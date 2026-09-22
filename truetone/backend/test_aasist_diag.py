import os
import sys
import torch
import numpy as np
import soundfile as sf
import hashlib

base_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(base_dir, 'app'))

from app.detection.aasist import AasistDetector
import app.preprocessing.pipeline as pipeline

model_path = os.path.join(base_dir, "third_party/aasist/models/weights/AASIST-L.pth")
detector = AasistDetector(model_path)

# Hash of state dict
sd = torch.load(model_path, map_location='cpu')
keys_hash = hashlib.md5(str(sd.keys()).encode()).hexdigest()
print(f"State dict hash (keys): {keys_hash}")
# Hash some weights
some_weight = sd[list(sd.keys())[0]]
print(f"Weight sum of first layer: {some_weight.sum().item():.4f}")

audio, sr = sf.read("natural.wav")
# Process exactly like test bench
window_chunk = audio[:2*sr] # 2 seconds
processed = pipeline.preprocess_for_detection(window_chunk, sr, debug=True)

print("Calling AASIST score...")
score = detector.score(processed, 16000)
print(f"Score: {score}")

