import os
import sys
import torch
import soundfile as sf

base_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(base_dir, 'app'))
from app.detection.aasist import AasistDetector
import app.preprocessing.pipeline as pipeline

detector = AasistDetector(os.path.join(base_dir, "third_party/aasist/models/weights/AASIST-L.pth"))
audio, sr = sf.read("synthetic.wav")
window_chunk = audio[:2*sr]
processed = pipeline.preprocess_for_detection(window_chunk, sr, debug=False)
score = detector.score(processed, 16000)
print(f"Synthetic Score: {score}")
