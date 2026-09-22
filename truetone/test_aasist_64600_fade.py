import sys
import numpy as np
import librosa
sys.path.append("backend")
from app.preprocessing.audio import preprocess_audio
from app.detection.aasist import AasistDetector

det = AasistDetector("backend/third_party/aasist/models/weights/AASIST-L.pth", "cpu")

audio_data, sr = librosa.load(librosa.ex('libri1'), sr=None)
audio_16k = librosa.resample(audio_data, orig_sr=sr, target_sr=16000)

window_samples = 64600
step_samples = int(1.0 * 16000)

fade_len = int(0.1 * 16000) # 100ms fade to be very smooth
fade_in = np.linspace(0, 1, fade_len)
fade_out = np.linspace(1, 0, fade_len)

start_idx = 0
while start_idx + window_samples <= len(audio_16k):
    window_chunk = audio_16k[start_idx:start_idx + window_samples].copy()
    
    # Apply fade
    window_chunk[:fade_len] *= fade_in
    window_chunk[-fade_len:] *= fade_out
    
    processed = preprocess_audio(window_chunk, 16000, target_sr=16000, vad_top_db=None)
    a_score = det.score(processed, 16000)
    
    print(f"Window {start_idx // step_samples}: Score = {a_score:.4f}")
    
    start_idx += step_samples

