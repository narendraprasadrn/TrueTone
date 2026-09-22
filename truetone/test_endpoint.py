import requests
import librosa
import soundfile as sf
import json

y, sr = librosa.load(librosa.ex('libri1'), sr=16000)
sf.write("test_libri.wav", y, sr)

with open("test_libri.wav", "rb") as f:
    files = {"file": ("test_libri.wav", f, "audio/wav")}
    res = requests.post("http://localhost:8000/test/analyze-audio", files=files)
    
data = res.json()
print("Overall:", data.get("overall_score"), data.get("overall_classification"))
for w in data.get("windows", []):
    print(f"Window {w['window_index']}: AASIST = {w['aasist_score']:.4f}, Fused = {w['fused_score']:.4f}")
