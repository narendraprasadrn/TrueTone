import sys
import torch
import torch.nn.functional as F
import numpy as np
import librosa

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

with open("backend/third_party/aasist/config/AASIST-L.conf", "r") as f:
    config = json.load(f)
model = Model(config["model_config"]).to("cpu")
model.load_state_dict(torch.load("backend/third_party/aasist/models/weights/AASIST-L.pth", map_location="cpu"))
model.eval()

y, sr = librosa.load(librosa.ex('libri1'), sr=16000)
y = y / np.abs(y).max()

for db in [60, 40, 30, 20, 10]:
    y_trim, _ = librosa.effects.trim(y, top_db=db)
    padded = pad(y_trim)
    x_inp = torch.Tensor(padded).unsqueeze(0)
    with torch.no_grad():
        _, out = model(x_inp)
        probs = F.softmax(out, dim=1)
        print(f"top_db={db}, length={len(y_trim)}, Score={float(probs[0, 0].numpy()):.4f}")

