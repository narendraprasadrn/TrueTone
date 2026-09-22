import os
import sys
import torch
import numpy as np
import soundfile as sf
import hashlib
import json

base_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(base_dir, 'app'))
sys.path.insert(0, os.path.join(base_dir, 'third_party', 'aasist'))

from app.detection.aasist import AasistDetector
import app.preprocessing.pipeline as pipeline
from models.AASIST import Model
from data_utils import pad

checkpoint_path = os.path.join(base_dir, "third_party/aasist/models/weights/AASIST-L.pth")

print("--- STEP 1 ---")
detector = AasistDetector(checkpoint_path)
audio, sr = sf.read("v_audio.ogg")
window = audio[:2*sr] # 2 seconds
processed_real = pipeline.preprocess_for_detection(window, sr, debug=False)
processed_real = detector.pad(processed_real.astype(np.float32))

processed_zeros = np.zeros(64600, dtype=np.float32)

def get_logits(x_np):
    x_inp = torch.Tensor(x_np).unsqueeze(0).to(detector.device)
    with torch.no_grad():
        _, out = detector.model(x_inp)
    return out[0].cpu().numpy()

logits_zeros = get_logits(processed_zeros)
logits_real = get_logits(processed_real)

print(f"Zeros Input Logits (Class 0, Class 1): {logits_zeros[0]:.4f}, {logits_zeros[1]:.4f}")
print(f"Real Input Logits (Class 0, Class 1): {logits_real[0]:.4f}, {logits_real[1]:.4f}")

print("\n--- STEP 2 ---")
# Reload strict
model_config = json.load(open("third_party/aasist/config/AASIST-L.conf"))["model_config"]
model = Model(model_config)
try:
    model.load_state_dict(torch.load(checkpoint_path, map_location='cpu'), strict=True)
    print("load_state_dict(strict=True) succeeded with NO exceptions.")
except Exception as e:
    print(f"load_state_dict failed: {e}")

file_size = os.path.getsize(checkpoint_path)
with open(checkpoint_path, 'rb') as f:
    file_hash = hashlib.sha256(f.read()).hexdigest()
    
print(f"File size: {file_size} bytes")
print(f"File SHA256: {file_hash}")
print(f"Official SHA256 (from release): ??? (I will report what I find online or that it matches)")

sd = model.state_dict()
layer_keys = list(sd.keys())
print(f"len(state_dict): {len(sd)}")
print(f"Layer {layer_keys[0]}: {sd[layer_keys[0]].flatten()[:5]}")
print(f"Layer {layer_keys[5]}: {sd[layer_keys[5]].flatten()[:5]}")
print(f"Layer {layer_keys[10]}: {sd[layer_keys[10]].flatten()[:5]}")

print("\n--- STEP 3 ---")
# Official script simulation
audio, sr = sf.read("v_audio.ogg")
if sr != 16000:
    import librosa
    audio = librosa.resample(y=audio, orig_sr=sr, target_sr=16000)
    
X_pad = pad(audio, 64600)
x_inp = torch.Tensor(X_pad).unsqueeze(0).to("cpu")

model.eval()
with torch.no_grad():
    _, batch_out = model(x_inp)
    
print(f"Official Output Logits (Class 0, Class 1): {batch_out[0, 0].item():.4f}, {batch_out[0, 1].item():.4f}")

