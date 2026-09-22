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
y = y[:48000] # 3 seconds

# Pass WITHOUT padding!
x_inp = torch.Tensor(y).unsqueeze(0)
with torch.no_grad():
    _, out = model(x_inp)
    probs = F.softmax(out, dim=1)
    print("No Padding 3s Score:", float(probs[0, 0].numpy()))

