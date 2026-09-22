import sys
import numpy as np
sys.path.append("backend")
from app.detection.aasist import AasistDetector

det = AasistDetector("third_party/aasist/models/weights/AASIST-L.pth")

# silence
sil = np.zeros(48000, dtype=np.float32)
score = det.score(sil, 16000)
print(f"AASIST silence score: {score}")

# noise (which is what silence becomes if normalized by max_val when max_val > 0)
noise = np.random.randn(48000).astype(np.float32)
noise = noise / np.abs(noise).max()
score2 = det.score(noise, 16000)
print(f"AASIST noise score: {score2}")

