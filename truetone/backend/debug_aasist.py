import os
import scipy.io.wavfile as wav
import librosa
import numpy as np
from app.preprocessing.pipeline import preprocess_for_detection, prepare_aasist_context
from app.detection.aasist import AasistDetector

base_dir = os.path.dirname(os.path.abspath(__file__))
aasist = AasistDetector(os.path.join(base_dir, "third_party/aasist/models/weights/AASIST-L.pth"))

audio_path = os.path.join(os.path.dirname(base_dir), "test_libri.wav")
y, sr = librosa.load(audio_path, sr=None)
y_proc = preprocess_for_detection(y, sr)

print(f"Total audio length: {len(y_proc)} samples ({len(y_proc)/16000:.2f}s)")

os.makedirs("truetone/backend/debug", exist_ok=True)

y_2s = y_proc[:32000]
context_2s = prepare_aasist_context(y_2s)

wav.write("truetone/backend/debug/testbench_aasist_window_2s_padded.wav", 16000, context_2s)

res_padded = aasist.predict_aasist(context_2s, 16000)
print(f"Padded 2s spoof score: {res_padded['spoof_score']:.4f}")

if len(y_proc) >= 64600:
    context_4s = y_proc[:64600]
    wav.write("truetone/backend/debug/testbench_aasist_window.wav", 16000, context_4s)
    res_4s = aasist.predict_aasist(context_4s, 16000)
    print(f"Contiguous 4s spoof score: {res_4s['spoof_score']:.4f}")

