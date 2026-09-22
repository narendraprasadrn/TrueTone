import sys
import numpy as np
sys.path.append("backend")
from app.preprocessing.audio import preprocess_audio

audio = np.random.randn(48000).astype(np.float32) * 0.05
res = preprocess_audio(audio, 16000, 16000, None)
print("Length post live:", len(res))

res2 = preprocess_audio(audio, 16000, 16000, 20)
print("Length post regular:", len(res2))
