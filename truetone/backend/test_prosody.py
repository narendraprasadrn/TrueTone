import sys
sys.path.append("backend")
import numpy as np
from app.detection.prosody import ProsodyDetector

window = np.random.randn(48000).astype(np.float32)
det = ProsodyDetector()
score = det.score(window)
print(f"Score: {score}")
