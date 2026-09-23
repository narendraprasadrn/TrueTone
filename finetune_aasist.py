import pandas as pd
import numpy as np
import os
import sys
import glob
import json
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset, Audio
import librosa
import datetime
from sklearn.metrics import roc_auc_score
import warnings
import random
warnings.filterwarnings("ignore")

sys.path.append(os.path.abspath("truetone/backend"))
from app.detection.aasist import AasistDetector
from third_party.aasist.models.AASIST import Model

def set_seed(seed=42):
    np.random.seed(seed)
    torch.manual_seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

set_seed(42)

# --- 1. Load Data Manifests ---
tamil_train = pd.read_csv("data/splits_tamil_hindi/tamil_train.csv")
hindi_train = pd.read_csv("data/splits_tamil_hindi/hindi_train.csv")
replay_train = pd.read_csv("data/splits_tamil_hindi/multilingual_replay_train.csv")
train_df = pd.concat([tamil_train, hindi_train, replay_train]).sample(frac=1, random_state=42)

val_df = pd.read_csv("data/splits_multilingual/val.csv") # Odia validation set

print(f"Train samples: {len(train_df)}")
print(f"Val samples: {len(val_df)}")

# Create ID sets
train_ids = set(train_df['id'])
val_ids = set(val_df['id'])

all_df = pd.concat([train_df, val_df]).set_index('id')

# --- 2. HuggingFace Dataset Integration ---
cache_dir = os.path.expanduser("~/.cache/huggingface/hub/datasets--SherryT997--IndicTTS-Deepfake-Challenge-Data/snapshots/*/data/train-*.parquet")
all_files = glob.glob(cache_dir)
expected_shards = [0, 1, 3, 4, 6, 7, 9, 10, 12, 13, 15, 16, 18, 19, 21, 22, 24, 25, 27, 28, 30, 31, 33]
files_to_load = [f for f in all_files if int(os.path.basename(f).split('-')[1]) in expected_shards]

ds = load_dataset("parquet", data_files=files_to_load, split="train")
ds = ds.cast_column("audio", Audio(sampling_rate=16000))

# We will cache the needed audio in memory (since 1888 + 163 = ~2000 samples, ~250MB total)
print("Extracting audio to memory...")
audio_cache = {}
for item in ds:
    idx = item['id']
    if idx in train_ids or idx in val_ids:
        # Preprocess down to mono
        arr = item['audio']['array']
        if arr.ndim > 1: arr = arr.mean(axis=0)
        audio_cache[idx] = arr.flatten()

print(f"Cached {len(audio_cache)} files.")

# --- 3. Dataset & Augmentation ---
class AASISTDataset(Dataset):
    def __init__(self, ids_list, df, audio_cache, augment=False):
        self.ids = ids_list
        self.df = df
        self.audio_cache = audio_cache
        self.augment = augment
        self.target_len = 64600
        
    def __len__(self):
        return len(self.ids)
        
    def __getitem__(self, idx):
        item_id = self.ids[idx]
        audio = self.audio_cache[item_id]
        
        # Crop or Pad
        curr_len = len(audio)
        if curr_len > self.target_len:
            if self.augment:
                # Random crop
                start = random.randint(0, curr_len - self.target_len)
            else:
                # Center crop
                start = (curr_len - self.target_len) // 2
            audio = audio[start:start+self.target_len]
        elif curr_len < self.target_len:
            pad_total = self.target_len - curr_len
            pad_left = pad_total // 2
            pad_right = pad_total - pad_left
            audio = np.pad(audio, (pad_left, pad_right), 'constant')
            
        # Augmentation
        if self.augment:
            if random.random() < 0.2:
                # Additive noise
                noise = np.random.randn(*audio.shape) * 0.005
                audio = audio + noise
            if random.random() < 0.2:
                # Gain
                audio = audio * random.uniform(0.5, 1.5)
            if random.random() < 0.2:
                # Clipping
                thresh = random.uniform(0.5, 0.9)
                audio = np.clip(audio, -thresh, thresh)
                
        # labels: is_tts=1 -> spoof (class 0 in AASIST)
        # is_tts=0 -> bonafide (class 1 in AASIST)
        is_tts = self.df.loc[item_id, 'is_tts']
        label = 0 if is_tts == 1 else 1
        
        return torch.Tensor(audio).float(), torch.tensor(label).long()

train_dataset = AASISTDataset(list(train_ids), all_df, audio_cache, augment=True)
val_dataset = AASISTDataset(list(val_ids), all_df, audio_cache, augment=False)

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=0)
val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False, num_workers=0)

# --- 4. Model Setup ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

ckpt_path = os.path.abspath("truetone/backend/third_party/aasist/models/weights/AASIST-L.pth")
config_path = os.path.abspath("truetone/backend/third_party/aasist/config/AASIST-L.conf")
with open(config_path, "r") as f:
    config = json.load(f)
model = Model(config["model_config"]).to(device)
model.load_state_dict(torch.load(ckpt_path, map_location=device), strict=True)

criterion = nn.CrossEntropyLoss(weight=torch.FloatTensor([0.1, 0.9]).to(device))
# Weight for CrossEntropy: class 0 is spoof, class 1 is bonafide
# Often models are biased to say spoof or bonafide. We use no weights first.
criterion = nn.CrossEntropyLoss()

optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5, weight_decay=1e-4)

# --- 5. Training Loop ---
epochs = 10
best_val_loss = float('inf')
patience = 4
patience_counter = 0

os.makedirs("models/aasist/finetuned_tamil_hindi", exist_ok=True)
best_model_path = "models/aasist/finetuned_tamil_hindi/best.pt"

print("Starting Fine-tuning...")
for epoch in range(epochs):
    model.train()
    train_loss = 0
    correct = 0
    total = 0
    
    for i, (audio, labels) in enumerate(train_loader):
        audio, labels = audio.to(device), labels.to(device)
        
        optimizer.zero_grad()
        _, logits = model(audio)
        
        loss = criterion(logits, labels)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        
        train_loss += loss.item()
        preds = logits.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)
        
    train_acc = correct / total
    
    # Validation
    model.eval()
    val_loss = 0
    val_correct = 0
    val_total = 0
    with torch.no_grad():
        for audio, labels in val_loader:
            audio, labels = audio.to(device), labels.to(device)
            _, logits = model(audio)
            loss = criterion(logits, labels)
            val_loss += loss.item()
            preds = logits.argmax(dim=1)
            val_correct += (preds == labels).sum().item()
            val_total += labels.size(0)
            
    val_acc = val_correct / val_total
    
    print(f"Epoch {epoch+1}/{epochs} | Train Loss: {train_loss/len(train_loader):.4f} Acc: {train_acc:.4f} | Val Loss: {val_loss/len(val_loader):.4f} Acc: {val_acc:.4f}")
    
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        torch.save(model.state_dict(), best_model_path)
        patience_counter = 0
        print(f" -> Saved best model")
    else:
        patience_counter += 1
        if patience_counter >= patience:
            print("Early stopping!")
            break

print("Training finished.")

# --- 6. Save Config Summary ---
config_summary = {
    'pretrained_checkpoint': ckpt_path,
    'best_model': best_model_path,
    'learning_rate': 2e-5,
    'weight_decay': 1e-4,
    'batch_size': 16,
    'epochs_run': epoch+1,
    'best_val_loss': best_val_loss
}
os.makedirs("config", exist_ok=True)
with open("config/train_aasist_tamil_hindi.yaml", "w") as f:
    json.dump(config_summary, f, indent=2)

