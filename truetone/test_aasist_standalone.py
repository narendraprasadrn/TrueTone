import sys
import os
import torch
import torch.nn.functional as F
import numpy as np
import librosa
from pathlib import Path

# Add third_party/aasist
sys.path.append("backend/third_party/aasist")
from models.AASIST import Model
import json

def pad(x, max_len=64600):
    x_len = x.shape[0]
    if x_len >= max_len:
        return x[:max_len]
    num_repeats = int(max_len / x_len) + 1
    padded_x = np.tile(x, (1, num_repeats))[:, :max_len][0]
    return padded_x

# Load model
config_path = "backend/third_party/aasist/config/AASIST-L.conf"
with open(config_path, "r") as f:
    config = json.load(f)
model = Model(config["model_config"]).to("cpu")
model.load_state_dict(torch.load("backend/third_party/aasist/models/weights/AASIST-L.pth", map_location="cpu"))
model.eval()

def score_audio(audio_array):
    padded = pad(audio_array)
    x_inp = torch.Tensor(padded).unsqueeze(0)
    with torch.no_grad():
        _, out = model(x_inp)
        probs = F.softmax(out, dim=1)
        # out[:, 1] is bonafide, out[:, 0] is spoof
        spoof_prob = float(probs[0, 0].numpy())
        bonafide_prob = float(probs[0, 1].numpy())
        raw_logits = out.numpy()
        return spoof_prob, bonafide_prob, raw_logits

# Generate synthetic audio
sr = 16000
t = np.linspace(0, 3, sr * 3, False)
synthetic = np.sin(440 * 2 * np.pi * t) * 0.5

# Generate naturalish audio
natural = np.random.randn(len(t)) * 0.05
natural += np.sin((440 + np.sin(t*10)*10) * 2 * np.pi * t) * 0.2

# Get actual audio if available
try:
    real_audio, _ = librosa.load(librosa.ex('trumpet'), sr=16000)
except:
    real_audio = natural

print("Synthetic:", score_audio(synthetic))
print("Natural:", score_audio(natural))
print("Real Trumpet:", score_audio(real_audio))
