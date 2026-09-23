import os
import sys
import torch
import numpy as np
import soundfile as sf
import json
import io

base_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(base_dir, 'third_party', 'aasist'))

from models.AASIST import Model
from data_utils import pad

import datasets

print("Loading dataset without auto-decoding...")
ds = datasets.load_dataset("SpeechAntiSpoofingBenchmarks/ASVspoof2019_LA", split="test", streaming=True)
ds = ds.cast_column("audio", datasets.Audio(decode=False))

bonafide_sample = None
for ex in ds:
    if ex.get("label") == 1 or ex.get("label") == "bonafide" or ex.get("is_spoof") == False:
        bonafide_sample = ex
        break

if not bonafide_sample:
    for ex in ds:
        print("Dataset element structure:", ex)
        if "label" in ex and (ex["label"] == 1 or ex["label"] == 'bonafide'):
            bonafide_sample = ex
            break

print("Bonafide sample found.")

audio_bytes = bonafide_sample["audio"]["bytes"]
audio_array, sr = sf.read(io.BytesIO(audio_bytes))

if sr != 16000:
    import librosa
    audio_array = librosa.resample(y=audio_array, orig_sr=sr, target_sr=16000)

audio_array = audio_array.astype(np.float32)
X_pad = pad(audio_array, 64600)
x_inp = torch.Tensor(X_pad).unsqueeze(0).to("cpu")

aasist_dir = os.path.abspath("third_party/aasist")
config_path = os.path.join(aasist_dir, "config", "AASIST-L.conf")
with open(config_path, "r") as f:
    config = json.loads(f.read())
model_config = config["model_config"]
model = Model(model_config).to("cpu")
model.load_state_dict(torch.load(os.path.join(aasist_dir, "models/weights/AASIST-L.pth"), map_location="cpu"))
model.eval()

with torch.no_grad():
    _, batch_out = model(x_inp)

print(f"HF Bonafide Sample Logits (Spoof=Class 0, Bonafide=Class 1): {batch_out[0,0].item():.4f}, {batch_out[0,1].item():.4f}")

