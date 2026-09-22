import sys
import numpy as np
import librosa
sys.path.append("backend")
from app.preprocessing.audio import preprocess_audio
from app.detection.aasist import AasistDetector

det = AasistDetector("backend/third_party/aasist/models/weights/AASIST-L.pth", "cpu")

audio_data, sr = librosa.load(librosa.ex('libri1'), sr=None)
print("Original SR:", sr)

window_duration_sec = 3.0
window_samples = int(window_duration_sec * sr)
step_samples = int(1.0 * sr)

start_idx = 0
total_samples = len(audio_data)

while start_idx + window_samples <= total_samples:
    end_idx = start_idx + window_samples
    window_chunk = audio_data[start_idx:end_idx]
    
    processed = preprocess_audio(window_chunk, sr, target_sr=16000)
    a_score = det.score(processed, 16000)
    
    print(f"Window {start_idx // step_samples}: Score = {a_score:.4f}")
    
    start_idx += step_samples

