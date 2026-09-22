import sys
import torch
import torch.nn.functional as F
import numpy as np
import librosa

sys.path.append("backend")
sys.path.append("backend/third_party/aasist")
from models.AASIST import Model
import json

def pad_front(x, max_len=64600):
    # Add 0.5s silence at the start
    silence = np.zeros(8000, dtype=x.dtype)
    x = np.concatenate([silence, x])
    x_len = x.shape[0]
    if x_len >= max_len:
        return x[:max_len]
    return np.pad(x, (0, max_len - x_len))

with open("backend/third_party/aasist/config/AASIST-L.conf", "r") as f:
    config = json.load(f)
model = Model(config["model_config"]).to("cpu")
model.load_state_dict(torch.load("backend/third_party/aasist/models/weights/AASIST-L.pth", map_location="cpu"))
model.eval()

audio_data, sr = librosa.load(librosa.ex('libri1'), sr=None)
audio_16k = librosa.resample(audio_data, orig_sr=sr, target_sr=16000)
audio_16k /= np.abs(audio_16k).max()

window_samples = 48000
step_samples = 16000
start_idx = 0

fade_len = int(0.05 * 16000) # 50ms
fade_in = np.linspace(0, 1, fade_len)
fade_out = np.linspace(1, 0, fade_len)

print("Testing FRONT PAD + FADE on 3s sliding windows")
while start_idx + window_samples <= len(audio_16k):
    window_chunk = audio_16k[start_idx:start_idx + window_samples].copy()
    
    window_chunk[:fade_len] *= fade_in
    window_chunk[-fade_len:] *= fade_out
    
    padded = pad_front(window_chunk)
    with torch.no_grad():
        out = F.softmax(model(torch.Tensor(padded).unsqueeze(0))[1], dim=1)[0, 0].numpy()
        
    print(f"Window {start_idx // step_samples}: Score = {out:.4f}")
    
    start_idx += step_samples

