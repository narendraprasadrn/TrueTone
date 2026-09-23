import os
import scipy.io.wavfile as wav
import librosa
import numpy as np
from app.preprocessing.pipeline import preprocess_for_detection, prepare_aasist_context
from app.detection.aasist import AasistDetector

base_dir = os.path.dirname(os.path.abspath(__file__))
aasist = AasistDetector(os.path.join(base_dir, "third_party/aasist/models/weights/AASIST-L.pth"))

audio_path = os.path.join(base_dir, "real_whatsapp.ogg")
y, sr = librosa.load(audio_path, sr=None)

print("=== DECODING INFO ===")
print(f"Sample Rate: {sr}")
print(f"Channels: {y.shape[0] if y.ndim > 1 else 1}")
print(f"Duration: {librosa.get_duration(y=y, sr=sr):.2f}s")
print(f"Format: OGG")

y_proc = preprocess_for_detection(y, sr)
print(f"Decoded PCM length: {len(y_proc)}")

# First window (0.0 to 4.0375s)
context_1 = y_proc[:64600]
res_1 = aasist.predict_aasist(context_1, 16000)

print("\n=== WINDOW 1 ===")
print(f"start: 0.0s")
print(f"end: 4.0375s")
print(f"samples: {len(context_1)}")
print(f"logits: {res_1['logits']}")
print(f"spoof_score: {res_1['spoof_score']:.4f}")
print(f"bonafide_score: {res_1['bonafide_score']:.4f}")

# Second window (1.0 to 5.0375s)
context_2 = y_proc[16000 : 16000 + 64600]
res_2 = aasist.predict_aasist(context_2, 16000)

print("\n=== WINDOW 2 ===")
print(f"start: 1.0s")
print(f"end: 5.0375s")
print(f"samples: {len(context_2)}")
print(f"logits: {res_2['logits']}")
print(f"spoof_score: {res_2['spoof_score']:.4f}")
print(f"bonafide_score: {res_2['bonafide_score']:.4f}")

