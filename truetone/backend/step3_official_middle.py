import os
import sys
import torch
import soundfile as sf
import json

aasist_dir = os.path.abspath("third_party/aasist")
sys.path.insert(0, aasist_dir)
from models.AASIST import Model
from data_utils import pad

config_path = os.path.join(aasist_dir, "config", "AASIST-L.conf")
with open(config_path, "r") as f:
    config = json.loads(f.read())
model_config = config["model_config"]
model = Model(model_config).to("cpu")
model.load_state_dict(torch.load(os.path.join(aasist_dir, "models/weights/AASIST-L.pth"), map_location="cpu"))
model.eval()

audio, sr = sf.read("v_audio.ogg")
print(f"Total length: {len(audio)/sr:.2f}s")

# Let's test 5 different 4-second chunks
with torch.no_grad():
    for start_s in [0, 10, 20, 30, 40]:
        start_idx = start_s * sr
        chunk = audio[start_idx:start_idx+64600]
        if len(chunk) < 64600:
            chunk = pad(chunk, 64600)
            
        x_inp = torch.Tensor(chunk).unsqueeze(0)
        _, batch_out = model(x_inp)
        print(f"Chunk starting at {start_s}s: Logits (spoof={batch_out[0,0].item():.4f}, bonafide={batch_out[0,1].item():.4f})")

