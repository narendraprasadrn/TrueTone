import sys
import numpy as np
sys.path.append("backend")
from app.detection.aasist import AasistDetector

det = AasistDetector("third_party/aasist/models/weights/AASIST-L.pth")

quiet_noise = np.random.randn(48000).astype(np.float32) * 0.0001
score = det.score(quiet_noise, 16000)
print(f"AASIST quiet noise score: {score}")
