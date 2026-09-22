import sys
sys.path.append("backend")
import numpy as np
from app.detection.prosody_features import extract_prosody_features

window = np.zeros(48000, dtype=np.float32)
features = extract_prosody_features(window)
print(f"Zeros input features: {features}")
