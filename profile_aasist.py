import torch
import time
import sys
import os

sys.path.append(os.path.abspath("truetone/backend"))
from third_party.aasist.models.AASIST import Model
import json
import numpy as np

print("--- DEVICE CHECK ---")
print(f"CUDA Available: {torch.cuda.is_available()}")
print(f"CUDA Device Count: {torch.cuda.device_count()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
else:
    print("GPU: None")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Target Device: {device}")

print("\n--- MODEL CHECK ---")
config_path = os.path.abspath("truetone/backend/third_party/aasist/config/AASIST-L.conf")
with open(config_path, "r") as f:
    config = json.load(f)
model = Model(config["model_config"]).to(device)

total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Total Params: {total_params}")
print(f"Trainable Params: {trainable_params}")

batch_size = 16
audio_len = 64600

print(f"\n--- PROFILING LOOP ---")
print(f"Batch Size: {batch_size}, Input Length: {audio_len}")

# Dummy Dataloader simulation
audio_tensor = torch.randn(batch_size, audio_len).to(device)
labels = torch.randint(0, 2, (batch_size,)).to(device)

criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5, weight_decay=1e-4)

model.train()

# Warmup
with torch.no_grad():
    _ = model(audio_tensor)

print("\nStarting timing for 5 batches...")
times = {'data': 0, 'forward': 0, 'backward': 0, 'optim': 0}

for i in range(5):
    # Dataloader simulation
    t0 = time.time()
    audio = torch.randn(batch_size, audio_len).to(device)
    lbls = torch.randint(0, 2, (batch_size,)).to(device)
    t1 = time.time()
    times['data'] += (t1 - t0)
    
    # Forward
    optimizer.zero_grad()
    _, logits = model(audio)
    loss = criterion(logits, lbls)
    t2 = time.time()
    times['forward'] += (t2 - t1)
    
    # Backward
    loss.backward()
    t3 = time.time()
    times['backward'] += (t3 - t2)
    
    # Optim
    optimizer.step()
    t4 = time.time()
    times['optim'] += (t4 - t3)

print("\n--- TIMING SUMMARY (Average per batch) ---")
print(f"Data Prep:  {times['data']/5:.4f}s")
print(f"Forward:    {times['forward']/5:.4f}s")
print(f"Backward:   {times['backward']/5:.4f}s")
print(f"Optim Step: {times['optim']/5:.4f}s")
print(f"Total/Batch:{sum(times.values())/5:.4f}s")

# CPU info
print("\n--- SYSTEM INFO ---")

