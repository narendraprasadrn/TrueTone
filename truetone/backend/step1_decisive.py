import os
import sys
import torch
import soundfile as sf
import json

aasist_dir = os.path.abspath("third_party/aasist")
sys.path.insert(0, aasist_dir)
from models.AASIST import Model
from data_utils import pad

# 1. Convert OGG to WAV
audio, sr = sf.read("real_whatsapp.ogg")
if sr != 16000:
    import librosa
    audio = librosa.resample(y=audio, orig_sr=sr, target_sr=16000)
    sr = 16000
sf.write("real_whatsapp_uncompressed.wav", audio, sr, subtype='PCM_16')

# 2. Run model
config_path = os.path.join(aasist_dir, "config", "AASIST-L.conf")
with open(config_path, "r") as f:
    config = json.loads(f.read())
model_config = config["model_config"]
model = Model(model_config).to("cpu")
model.load_state_dict(torch.load(os.path.join(aasist_dir, "models/weights/AASIST-L.pth"), map_location="cpu"))
model.eval()

audio_wav, sr_wav = sf.read("real_whatsapp_uncompressed.wav")

with torch.no_grad():
    for i in range(4): # First 4 windows
        start_idx = i * 16000
        chunk = audio_wav[start_idx:start_idx+64600]
        if len(chunk) < 64600:
            chunk = pad(chunk, 64600)
        x_inp = torch.Tensor(chunk).unsqueeze(0)
        _, batch_out = model(x_inp)
        print(f"WAV Window {i}: Logits (spoof={batch_out[0,0].item():.4f}, bonafide={batch_out[0,1].item():.4f})")

