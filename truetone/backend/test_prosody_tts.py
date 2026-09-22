import sys
import os
sys.path.append("backend")
import numpy as np
import librosa
from app.detection.prosody import ProsodyDetector

# generate a 3-second 440Hz sine wave (very synthetic)
sr = 16000
t = np.linspace(0, 3, sr * 3, False)
y = np.sin(440 * 2 * np.pi * t) * 0.5
y = y.astype(np.float32)

det = ProsodyDetector()
score = det.score(y, sr)
print(f"Final Prosody Score for synthetic sine wave: {score}")

# generate a somewhat natural sound (noise + harmonics + silence)
y_natural = np.sin(440 * 2 * np.pi * t) * 0.2 + np.random.randn(len(t))*0.05
y_natural[:16000] = 0 # 1s silence
y_natural = y_natural.astype(np.float32)

score_nat = det.score(y_natural, sr)
print(f"Final Prosody Score for naturalish wave: {score_nat}")
