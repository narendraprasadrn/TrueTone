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
    padded_x = np.tile(x, (num_repeats,))[:max_len]
    return padded_x

with open("backend/third_party/aasist/config/AASIST-L.conf", "r") as f:
    config = json.load(f)
model = Model(config["model_config"]).to("cpu")
model.load_state_dict(torch.load("backend/third_party/aasist/models/weights/AASIST-L.pth", map_location="cpu"))
model.eval()

y, sr = librosa.load(librosa.ex('libri1'), sr=16000)
y = y / np.abs(y).max()

# Take the first 3 seconds (48000 samples)
y_3s = y[:48000]

padded = pad(y_3s)
x_inp = torch.Tensor(padded).unsqueeze(0)
with torch.no_grad():
    _, out = model(x_inp)
    probs = F.softmax(out, dim=1)
    print("3s tiled to 64600 Score:", float(probs[0, 0].numpy()))

# What if we just take 64600 samples without tiling?
y_4s = y[:64600]
x_inp_4s = torch.Tensor(y_4s).unsqueeze(0)
with torch.no_grad():
    _, out2 = model(x_inp_4s)
    probs2 = F.softmax(out2, dim=1)
    print("4s native Score:", float(probs2[0, 0].numpy()))

