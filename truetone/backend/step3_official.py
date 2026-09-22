import os
import sys
import torch
import numpy as np
import soundfile as sf
import json
from pathlib import Path

# Setup path like main.py
aasist_dir = os.path.abspath("third_party/aasist")
sys.path.insert(0, aasist_dir)

from models.AASIST import Model
from data_utils import pad

# Load model
config_path = os.path.join(aasist_dir, "config", "AASIST-L.conf")
with open(config_path, "r") as f:
    config = json.loads(f.read())

model_config = config["model_config"]
model = Model(model_config).to("cpu")
model.load_state_dict(torch.load(os.path.join(aasist_dir, "models/weights/AASIST-L.pth"), map_location="cpu"))
model.eval()

# Load v_audio.ogg exactly like dataset loader
# Dataset uses sf.read and then pad
audio, sr = sf.read("v_audio.ogg")

# Take the first 64600 samples or pad
if sr != 16000:
    print(f"Warning: SR is {sr}, official repo expects 16000")
    
# Let's just process the first chunk exactly like the dataset class
X_pad = pad(audio, 64600)
x_inp = torch.Tensor(X_pad).unsqueeze(0)

with torch.no_grad():
    _, batch_out = model(x_inp)
    
print(f"Official Output Logits (class 0, class 1): {batch_out[0, 0].item():.4f}, {batch_out[0, 1].item():.4f}")
print(f"Official 'Score' (batch_out[:, 1]): {batch_out[0, 1].item():.4f}")

