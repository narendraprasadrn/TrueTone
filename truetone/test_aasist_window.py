import sys
import numpy as np
import librosa
sys.path.append("backend")
from app.preprocessing.audio import preprocess_audio
from app.detection.aasist import AasistDetector

det = AasistDetector("backend/third_party/aasist/models/weights/AASIST-L.pth", "cpu")

audio_data, sr = librosa.load(librosa.ex('libri1'), sr=None)
print("Original SR:", sr)

window_samples = 64600 # Exactly what AASIST expects!
step_samples = int(1.0 * 16000)

start_idx = 0
# Resample the entire audio to 16000 FIRST so windowing is clean
audio_16k = librosa.resample(audio_data, orig_sr=sr, target_sr=16000)
total_samples = len(audio_16k)

while start_idx + window_samples <= total_samples:
    end_idx = start_idx + window_samples
    window_chunk = audio_16k[start_idx:end_idx]
    
    # Preprocess but SKIP VAD (VAD changes length!)
    processed = preprocess_audio(window_chunk, 16000, target_sr=16000, vad_top_db=None)
    
    a_score = det.score(processed, 16000)
    
    print(f"Window {start_idx // step_samples}: Length={len(processed)}, Score = {a_score:.4f}")
    
    start_idx += step_samples

