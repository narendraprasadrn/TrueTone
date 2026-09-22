import librosa
import numpy as np

# Ambient noise (RMS ~0.001, max ~0.005)
noise = np.random.randn(48000).astype(np.float32) * 0.001

# Try to trim
voiced, index = librosa.effects.trim(noise, top_db=20)
print(f"Noise length after trim: {len(voiced)} from {len(noise)}")

# Speech with noise
speech = np.random.randn(48000).astype(np.float32) * 0.001
speech[20000:30000] += np.sin(np.linspace(0, 1000, 10000)) * 0.1
voiced_sp, index_sp = librosa.effects.trim(speech, top_db=20)
print(f"Speech length after trim: {len(voiced_sp)} from {len(speech)}")
