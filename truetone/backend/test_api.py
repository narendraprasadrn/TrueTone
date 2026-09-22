import requests
import numpy as np
import librosa
import soundfile as sf
import time

sr = 16000
t = np.linspace(0, 6, sr * 6, False)
# Synthetic: perfect pitch, no jitter, no shimmer, no noise
synthetic = np.sin(440 * 2 * np.pi * t) * 0.5
synthetic = synthetic.astype(np.float32)
sf.write("synthetic.wav", synthetic, sr)

# Natural: noisy, varying pitch, pauses
# We'll use a real audio sample from librosa if available, otherwise construct one
try:
    natural, _ = librosa.load(librosa.ex('trumpet'), sr=16000, duration=6.0)
except Exception:
    natural = np.sin((440 + np.sin(t*10)*10) * 2 * np.pi * t) * 0.2 + np.random.randn(len(t))*0.05
    natural[:16000] = 0 # pause
    natural = natural.astype(np.float32)
sf.write("natural.wav", natural, sr)

# Test synthetic
with open("synthetic.wav", "rb") as f:
    res = requests.post("http://localhost:8000/test/analyze-audio", files={"file": f})
    print("Synthetic Overall Class:", res.json().get("overall_classification"))
    print("Synthetic Overall Score:", res.json().get("overall_score"))
    for w in res.json().get("windows", []):
        print("  Win", w["window_index"], "Prosody:", w["prosody_score"], "Fused:", w["fused_score"])

# Test natural
with open("natural.wav", "rb") as f:
    res = requests.post("http://localhost:8000/test/analyze-audio", files={"file": f})
    print("Natural Overall Class:", res.json().get("overall_classification"))
    print("Natural Overall Score:", res.json().get("overall_score"))
    for w in res.json().get("windows", []):
        print("  Win", w["window_index"], "Prosody:", w["prosody_score"], "Fused:", w["fused_score"])
