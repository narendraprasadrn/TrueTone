import pandas as pd
import numpy as np
import os
import sys
import glob
import json
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import random
import warnings
import time

warnings.filterwarnings("ignore")
sys.path.append(os.path.abspath("truetone/backend"))
from app.detection.aasist import AasistDetector
from third_party.aasist.models.AASIST import Model
from fix_loading import load_audio_fast

def set_seed(seed=42):
    np.random.seed(seed)
    torch.manual_seed(seed)
    random.seed(seed)

set_seed(42)

# Ensure models dir exists
os.makedirs("models/aasist/finetuned_multilingual", exist_ok=True)
os.makedirs("reports/aasist", exist_ok=True)

print("1. DATASET / SPLITS")
train_full = pd.read_csv("data/splits_multilingual/internal_train.csv")
val_full = pd.read_csv("data/splits_multilingual/internal_val.csv")
test_df = pd.read_csv("data/splits_multilingual/test.csv")

# Use full datasets as requested for actual training
train_df = train_full
val_df = val_full

print(f"Train samples: {len(train_df)}")
print(f"Val samples: {len(val_df)}")
print(f"Test samples: {len(test_df)}")

train_ids = set(train_df['id'])
val_ids = set(val_df['id'])
test_ids = set(test_df['id'])

all_df = pd.concat([train_df, val_df, test_df]).set_index('id')

print("Loading audio into memory...")
needed_ids = train_ids | val_ids | test_ids
audio_cache = load_audio_fast(needed_ids)
print(f"Cached {len(audio_cache)} files out of {len(needed_ids)} needed.")

class AASISTDataset(Dataset):
    def __init__(self, ids_list, df, audio_cache, augment=False):
        # Only keep ids that were successfully loaded
        self.ids = [i for i in ids_list if i in audio_cache]
        self.df = df
        self.audio_cache = audio_cache
        self.augment = augment
        self.target_len = 64600
        
    def __len__(self):
        return len(self.ids)
        
    def __getitem__(self, idx):
        item_id = self.ids[idx]
        audio = self.audio_cache[item_id]
        
        # Audio input (crop/pad)
        curr_len = len(audio)
        if curr_len > self.target_len:
            if self.augment:
                start = random.randint(0, curr_len - self.target_len)
            else:
                start = (curr_len - self.target_len) // 2
            audio = audio[start:start+self.target_len]
        elif curr_len < self.target_len:
            pad_total = self.target_len - curr_len
            pad_left = pad_total // 2
            pad_right = pad_total - pad_left
            audio = np.pad(audio, (pad_left, pad_right), 'constant')
            
        # Augmentation
        if self.augment:
            if random.random() < 0.3:
                # Additive noise
                noise = np.random.randn(*audio.shape) * random.uniform(0.001, 0.005)
                audio = audio + noise
            if random.random() < 0.3:
                # Gain
                audio = audio * random.uniform(0.5, 1.5)
            if random.random() < 0.1:
                # Clipping
                thresh = random.uniform(0.7, 0.9)
                audio = np.clip(audio, -thresh, thresh)
                
        is_tts = self.df.loc[item_id, 'is_tts']
        # class 0 = spoof, class 1 = bonafide (as in AASIST original)
        label = 0 if is_tts == 1 else 1
        
        return torch.Tensor(audio).float(), torch.tensor(label).long()

train_dataset = AASISTDataset(list(train_ids), all_df, audio_cache, augment=True)
val_dataset = AASISTDataset(list(val_ids), all_df, audio_cache, augment=False)
test_dataset = AASISTDataset(list(test_ids), all_df, audio_cache, augment=False)

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=0)
val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False, num_workers=0)
test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False, num_workers=0)

device = torch.device("cpu")
print(f"Using device: {device}")

ckpt_path = os.path.abspath("truetone/backend/third_party/aasist/models/weights/AASIST-L.pth")
config_path = os.path.abspath("truetone/backend/third_party/aasist/config/AASIST-L.conf")
with open(config_path, "r") as f:
    config = json.load(f)

def eval_model(model_obj, loader):
    model_obj.eval()
    all_preds = []
    all_labels = []
    all_spoof_scores = []
    with torch.no_grad():
        for audio, labels in loader:
            audio = audio.to(device)
            _, logits = model_obj(audio)
            probs = torch.softmax(logits, dim=1)
            # spoof score is probability of class 0
            spoof_scores = probs[:, 0].cpu().numpy()
            
            preds = logits.argmax(dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.cpu().numpy())
            all_spoof_scores.extend(spoof_scores)
            
    all_labels = np.array(all_labels)
    all_preds = np.array(all_preds)
    all_spoof_scores = np.array(all_spoof_scores)
    
    # We want labels to be 1 for spoof for AUC calculation, where spoof_score correlates with spoof
    y_true_spoof = (all_labels == 0).astype(int) 
    y_pred_spoof = (all_preds == 0).astype(int)
    
    acc = accuracy_score(all_labels, all_preds)
    auc = roc_auc_score(y_true_spoof, all_spoof_scores) if len(np.unique(y_true_spoof)) > 1 else None
    
    real_mean = all_spoof_scores[y_true_spoof == 0].mean() if np.any(y_true_spoof == 0) else None
    synth_mean = all_spoof_scores[y_true_spoof == 1].mean() if np.any(y_true_spoof == 1) else None
    
    return {
        "acc": float(acc),
        "auc": float(auc) if auc is not None else None,
        "real_mean": float(real_mean) if real_mean is not None else None,
        "synth_mean": float(synth_mean) if synth_mean is not None else None,
        "y_true_spoof": y_true_spoof,
        "y_pred_spoof": y_pred_spoof,
        "all_spoof_scores": all_spoof_scores,
        "all_preds": all_preds,
        "all_labels": all_labels
    }

# --- EVAL PRETRAINED ---
print("\n--- Evaluating Pretrained AASIST-L on Val ---")
model = Model(config["model_config"]).to(device)
model.load_state_dict(torch.load(ckpt_path, map_location=device), strict=True)
pretrained_val_res = eval_model(model, val_loader)
print(f"Pretrained Val Acc: {pretrained_val_res['acc']:.4f}, AUC: {pretrained_val_res['auc']:.4f}")
print(f"Pretrained Real mean spoof: {pretrained_val_res['real_mean']:.4f}, Synth mean spoof: {pretrained_val_res['synth_mean']:.4f}")

# --- TRAINING ---
print("\n--- Starting Training (Smoke test & Full) ---")
epochs = 4
optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5, weight_decay=1e-4)
criterion = nn.CrossEntropyLoss()

best_val_loss = float('inf')
best_model_path = "models/aasist/finetuned_multilingual/best_multilingual.pt"
patience = 2
patience_counter = 0
epoch_metrics = []

for epoch in range(epochs):
    model.train()
    train_loss = 0
    correct = 0
    total = 0
    start_time = time.time()
    
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
        
        if epoch == 0 and i == 0:
            print("Smoke test batch 0 passed. Forward/backward/loss functioning properly.")
        if epoch == 0 and i == 20:
            print("Smoke test 20 batches passed. Continuing full training...")
            
    train_acc = correct / total
    train_loss /= len(train_loader)
    
    # Validation
    val_res = eval_model(model, val_loader)
    
    # Calculate Val Loss
    model.eval()
    val_loss = 0
    with torch.no_grad():
        for audio, labels in val_loader:
            audio, labels = audio.to(device), labels.to(device)
            _, logits = model(audio)
            loss = criterion(logits, labels)
            val_loss += loss.item()
    val_loss /= len(val_loader)
    
    epoch_time = time.time() - start_time
    print(f"Epoch {epoch+1}/{epochs} ({epoch_time:.0f}s) | Train Loss: {train_loss:.4f} Acc: {train_acc:.4f} | Val Loss: {val_loss:.4f} Acc: {val_res['acc']:.4f} AUC: {val_res['auc']:.4f}")
    print(f"    Val Real spoof mean: {val_res['real_mean']:.4f} | Val Synth spoof mean: {val_res['synth_mean']:.4f}")
    
    epoch_metrics.append({
        "epoch": epoch+1,
        "train_loss": train_loss,
        "train_acc": train_acc,
        "val_loss": val_loss,
        "val_acc": val_res['acc'],
        "val_auc": val_res['auc'],
        "val_real_mean": val_res['real_mean'],
        "val_synth_mean": val_res['synth_mean']
    })
    
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        torch.save(model.state_dict(), best_model_path)
        patience_counter = 0
        print(f" -> Saved best model (loss: {val_loss:.4f})")
    else:
        patience_counter += 1
        if patience_counter >= patience:
            print(f"Early stopping at epoch {epoch+1}!")
            break

# --- LOCKED TEST EVALUATION ---
print("\n--- Evaluating Best Model on Locked Test ---")
model.load_state_dict(torch.load(best_model_path, map_location=device), strict=True)
test_res = eval_model(model, test_loader)

y_true_test = test_res['y_true_spoof']
y_pred_test = test_res['y_pred_spoof']
test_acc = accuracy_score(y_true_test, y_pred_test)
test_prec = precision_score(y_true_test, y_pred_test)
test_rec = recall_score(y_true_test, y_pred_test)
test_f1 = f1_score(y_true_test, y_pred_test)
test_conf = confusion_matrix(y_true_test, y_pred_test)

print(f"Locked Test Results:")
print(f"AUC: {test_res['auc']:.4f}")
print(f"Acc: {test_acc:.4f}")
print(f"Precision: {test_prec:.4f}")
print(f"Recall: {test_rec:.4f}")
print(f"F1: {test_f1:.4f}")
print(f"Confusion Matrix:\n{test_conf}")

# --- PER LANGUAGE EVALUATION ---
print("\n--- Per-Language Locked Test Results ---")
test_df_ids = test_dataset.ids
test_preds = test_res['y_pred_spoof']
test_labels = test_res['y_true_spoof']
test_scores = test_res['all_spoof_scores']

lang_results = {}
for i, item_id in enumerate(test_df_ids):
    lang = all_df.loc[item_id, 'language']
    if lang not in lang_results:
        lang_results[lang] = {'y_true': [], 'y_score': [], 'y_pred': []}
    lang_results[lang]['y_true'].append(test_labels[i])
    lang_results[lang]['y_score'].append(test_scores[i])
    lang_results[lang]['y_pred'].append(test_preds[i])

for lang, data in lang_results.items():
    if len(np.unique(data['y_true'])) > 1:
        l_auc = roc_auc_score(data['y_true'], data['y_score'])
    else:
        l_auc = float('nan')
    l_acc = accuracy_score(data['y_true'], data['y_pred'])
    print(f"  {lang}: AUC = {l_auc:.4f}, Acc = {l_acc:.4f} (N={len(data['y_true'])})")

# --- REAL-WORLD EXTERNAL CHECK ---
print("\n--- Real-World External Check ---")
import soundfile as sf
def eval_external_audio(path):
    if not os.path.exists(path):
        return f"File {path} not found"
    audio, sr = sf.read(path)
    if len(audio.shape) > 1: audio = audio.mean(axis=1)
    if sr != 16000:
        import librosa
        audio = librosa.resample(audio, orig_sr=sr, target_sr=16000)
    
    target_len = 64600
    curr_len = len(audio)
    if curr_len > target_len:
        audio = audio[:target_len]
    elif curr_len < target_len:
        audio = np.pad(audio, (0, target_len - curr_len), 'constant')
        
    x_inp = torch.Tensor(audio).float().unsqueeze(0).to(device)
    model.eval()
    with torch.no_grad():
        _, logits = model(x_inp)
        probs = torch.softmax(logits, dim=1)
        spoof_score = probs[0, 0].cpu().numpy()
        pred_class = "spoof" if spoof_score > 0.5 else "bonafide"
    return f"AASIST Score: {spoof_score:.4f}, Predicted: {pred_class}"

ext_files = [
    "truetone/backend/real_whatsapp.ogg",
    "truetone/backend/real_whatsapp_uncompressed.wav",
    "truetone/backend/natural.wav",
    "truetone/backend/synthetic.wav"
]
# Let's find real files in data/ or root
for f in ext_files:
    print(f"{f}: {eval_external_audio(f)}")

print("\nEvaluation complete. Saving metrics summary.")
with open("reports/aasist/domain_adapt_metrics.json", "w") as f:
    json.dump({
        "epoch_metrics": epoch_metrics,
        "pretrained_val": pretrained_val_res['auc'],
        "test_results": {
            "auc": test_res['auc'],
            "acc": test_acc,
            "f1": test_f1
        }
    }, f, indent=2)

print("\nAASIST-L DOMAIN ADAPTATION COMPLETE - VALIDATED WITHOUT PRODUCTION CHANGES.")
