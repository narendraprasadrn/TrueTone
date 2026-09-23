import torch
import time
import sys
import os

sys.path.append(os.path.abspath("truetone/backend"))
from third_party.aasist.models.AASIST import Model
import json
import numpy as np

device = torch.device("cpu")
config_path = os.path.abspath("truetone/backend/third_party/aasist/config/AASIST-L.conf")
with open(config_path, "r") as f:
    config = json.load(f)
model = Model(config["model_config"]).to(device)

print("Starting 1 forward pass...")
audio = torch.randn(1, 64600).to(device)
t0 = time.time()
_ = model(audio)
t1 = time.time()
print(f"Forward pass (bs=1) took: {t1-t0:.4f}s")

audio = torch.randn(16, 64600).to(device)
t0 = time.time()
_ = model(audio)
t1 = time.time()
print(f"Forward pass (bs=16) took: {t1-t0:.4f}s")
