import os
import sys
import torch
import soundfile as sf
import numpy as np

base_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(base_dir, 'app'))

from app.detection.prosody import ProsodyDetector
import app.preprocessing.pipeline as pipeline

detector = ProsodyDetector(os.path.join(base_dir, "app/detection/models/prosody_lgbm.txt"))

def score_audio(filepath):
    audio, sr = sf.read(filepath)
    if sr != 16000:
        import librosa
        audio = librosa.resample(y=audio, orig_sr=sr, target_sr=16000)
    
    # Let's check window 0 and 1
    for i in range(2):
        start = i * 16000 # 1 sec step
        window = audio[start:start+32000] # 2 sec window
        if len(window) < 32000: continue
        processed = pipeline.preprocess_for_detection(window, 16000, debug=False)
        score = detector.score(processed, 16000)
        print(f"  Window {i}: {score:.4f}")

print("Prosody: Real WhatsApp Audio (17.4s)")
score_audio("real_whatsapp.ogg")

print("\nProsody: Synthetic/Cloned Audio")
score_audio("synthetic_cloned.ogg")

