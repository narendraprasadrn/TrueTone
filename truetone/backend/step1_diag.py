import os
import sys
import torch
import numpy as np
import soundfile as sf
import json
from pathlib import Path

base_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(base_dir, 'app'))

from app.detection.aasist import AasistDetector
import app.preprocessing.pipeline as pipeline

detector = AasistDetector("third_party/aasist/models/weights/AASIST-L.pth")

# Get v_audio.ogg
audio, sr = sf.read("v_audio.ogg")
# Take first 2 seconds to match our window logic
window_chunk = audio[:2*sr]
processed_real = pipeline.preprocess_for_detection(window_chunk, sr, debug=False)
processed_real = processed_real.astype(np.float32)
processed_real_padded = detector.pad(processed_real)

# Create zeros
processed_zeros_padded = np.zeros(64600, dtype=np.float32)

print(f"Real input shape: {processed_real_padded.shape}, zeros input shape: {processed_zeros_padded.shape}")

# Run through model manually
def get_logits(x_np):
    x_inp = torch.Tensor(x_np).unsqueeze(0).to(detector.device)
    with torch.no_grad():
        _, out = detector.model(x_inp)
        probs = torch.nn.functional.softmax(out, dim=1)
    return out[0].cpu().numpy(), probs[0].cpu().numpy()

logits_zeros, probs_zeros = get_logits(processed_zeros_padded)
logits_real, probs_real = get_logits(processed_real_padded)

print("\n--- STEP 1 RESULTS ---")
print(f"Zeros Input Logits (class 0, class 1): {logits_zeros[0]:.4f}, {logits_zeros[1]:.4f}")
print(f"Zeros Input Probs  (class 0, class 1): {probs_zeros[0]:.4f}, {probs_zeros[1]:.4f}")
print(f"Real Input Logits  (class 0, class 1): {logits_real[0]:.4f}, {logits_real[1]:.4f}")
print(f"Real Input Probs   (class 0, class 1): {probs_real[0]:.4f}, {probs_real[1]:.4f}")

