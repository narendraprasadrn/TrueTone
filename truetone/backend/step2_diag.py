import os
import sys
import torch
import hashlib
from pathlib import Path

base_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(base_dir, 'app'))

from app.detection.aasist import AasistDetector
detector = AasistDetector("third_party/aasist/models/weights/AASIST-L.pth")

path = os.path.abspath("third_party/aasist/models/weights/AASIST-L.pth")
print(f"Absolute path opened: {path}")

# Load with strict=True and no exception handling
try:
    detector.model.load_state_dict(torch.load(path, map_location='cpu'), strict=True)
    print("load_state_dict(strict=True) succeeded with no exceptions.")
except Exception as e:
    print(f"load_state_dict failed: {e}")

sd = detector.model.state_dict()
print(f"len(state_dict): {len(sd)}")

# Print first 5 values of 3 different layers
layer_names = list(sd.keys())
print(f"Layer 1 ({layer_names[0]}): {sd[layer_names[0]].flatten()[:5]}")
print(f"Layer 5 ({layer_names[5]}): {sd[layer_names[5]].flatten()[:5]}")
print(f"Layer 10 ({layer_names[10]}): {sd[layer_names[10]].flatten()[:5]}")

print(f"Weight sum of layer 1: {sd[layer_names[0]].sum().item()}")

with open(path, 'rb') as f:
    file_hash = hashlib.md5(f.read()).hexdigest()
print(f"File MD5: {file_hash}")

