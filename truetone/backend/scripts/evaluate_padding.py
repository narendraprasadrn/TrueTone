import os
import sys
import numpy as np
import librosa
import warnings
warnings.filterwarnings("ignore")

base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(base_dir)

from app.preprocessing.pipeline import preprocess_for_detection
from app.detection.aasist import AasistDetector

detector = AasistDetector(os.path.join(base_dir, "third_party/aasist/models/weights/AASIST-L.pth"))

def pad_context(audio_buffer: np.ndarray, mode: str) -> np.ndarray:
    REQUIRED_SAMPLES = 64600
    if len(audio_buffer) >= REQUIRED_SAMPLES:
        return audio_buffer[:REQUIRED_SAMPLES]
    if mode == "zero":
        pad_width = REQUIRED_SAMPLES - len(audio_buffer)
        return np.pad(audio_buffer, (pad_width, 0), mode='constant', constant_values=0.0)
    elif mode == "tile":
        num_repeats = int(REQUIRED_SAMPLES / len(audio_buffer)) + 1
        return np.tile(audio_buffer, (1, num_repeats))[:, :REQUIRED_SAMPLES][0]

def analyze_file(file_path: str, mode: str, gain: float = 1.0) -> dict:
    y, sr = librosa.load(file_path, sr=16000)
    if gain != 1.0:
        y = np.clip(y * gain, -1.0, 1.0)
    
    step_samples = int(1.0 * sr)
    window_samples = int(2.0 * sr)
    total_samples = len(y)
    
    scores = []
    
    # Process only first 5 windows to speed up
    for i in range(0, min(total_samples, step_samples * 5), step_samples):
        other_start = i
        other_end = min(total_samples, other_start + window_samples)
        if other_end - other_start < step_samples:
            break
            
        aasist_start = other_start
        aasist_end = min(total_samples, aasist_start + 64600)
        aasist_chunk = y[aasist_start:aasist_end]
        
        proc = preprocess_for_detection(aasist_chunk, sr)
        ctx = pad_context(proc, mode)
        
        res = detector.predict_aasist(ctx, 16000)
        scores.append(res['spoof_score'])
        
    return {
        "mean": np.mean(scores),
        "min": np.min(scores),
        "max": np.max(scores),
        "windows_above_50": sum(1 for s in scores if s > 0.5)
    }

print("\n--- 3. RE-MEASURE PADDING ---")
for file in ["v_audio.ogg", "synthetic_cloned.ogg", "real_whatsapp_uncompressed.wav"]:
    if os.path.exists(file):
        for mode in ["zero", "tile"]:
            res = analyze_file(file, mode)
            print(f"{file} ({mode}): mean={res['mean']:.4f}, min={res['min']:.4f}, max={res['max']:.4f}, >0.5={res['windows_above_50']}")

print("\n--- 4. GAIN SWEEP ON SPEECH (tile) ---")
gains = [0.05, 0.1, 0.25, 0.5, 1.0, 2.0]
for g in gains:
    res = analyze_file("real_whatsapp_uncompressed.wav", "tile", gain=g)
    print(f"Gain x{g}: mean spoof={res['mean']:.4f}")
