import os
import sys
import torch
import numpy as np
import soundfile as sf

base_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(base_dir, 'app'))

from app.detection.aasist import AasistDetector
detector = AasistDetector("third_party/aasist/models/weights/AASIST-L.pth")

# Load raw audio just like official repo
audio, sr = sf.read("v_audio.ogg")
if sr != 16000:
    import librosa
    audio = librosa.resample(y=audio, orig_sr=sr, target_sr=16000)

window = audio[:16000*2] # 2 seconds
window_padded = detector.pad(window)

x_inp = torch.Tensor(window_padded).unsqueeze(0).to(detector.device)
with torch.no_grad():
    _, out = detector.model(x_inp)

print(f"RAW AUDIO (No Preprocessing) Logits (spoof, bonafide): {out[0, 0].item():.4f}, {out[0, 1].item():.4f}")

# And what if we do 4 seconds?
window_4s = audio[:16000*4]
window_4s_padded = detector.pad(window_4s)
x_inp_4s = torch.Tensor(window_4s_padded).unsqueeze(0).to(detector.device)
with torch.no_grad():
    _, out_4s = detector.model(x_inp_4s)
print(f"RAW AUDIO (4 seconds) Logits (spoof, bonafide): {out_4s[0, 0].item():.4f}, {out_4s[0, 1].item():.4f}")

