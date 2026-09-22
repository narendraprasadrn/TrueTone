import sys
import numpy as np
import librosa
sys.path.append("backend")
from app.detection.aasist import AasistDetector

det = AasistDetector("backend/third_party/aasist/models/weights/AASIST-L.pth", "cpu")

y, sr = librosa.load(librosa.ex('libri1'), sr=16000)
score = det.score(y, sr)
print("AasistDetector LibriSpeech Score:", score)

