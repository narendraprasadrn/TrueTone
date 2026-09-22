import sys
import torch
import torch.nn.functional as F
import numpy as np
import librosa

sys.path.append("backend")
sys.path.append("backend/third_party/aasist")
from models.AASIST import Model
import json

with open("backend/third_party/aasist/config/AASIST-L.conf", "r") as f:
    config = json.load(f)
model = Model(config["model_config"]).to("cpu")
model.load_state_dict(torch.load("backend/third_party/aasist/models/weights/AASIST-L.pth", map_location="cpu"))
model.eval()

y, sr = librosa.load(librosa.ex('libri1'), sr=16000)

window_samples = 48000
step_samples = 16000
start_idx = 0

while start_idx + window_samples <= len(y):
    window_chunk = y[start_idx:start_idx + window_samples]
    
    x_inp = torch.Tensor(window_chunk).unsqueeze(0)
    with torch.no_grad():
        _, out = model(x_inp)
        probs = F.softmax(out, dim=1)
        
    print(f"Window {start_idx // step_samples}: Score = {float(probs[0, 0].numpy()):.4f}")
    
    start_idx += step_samples

