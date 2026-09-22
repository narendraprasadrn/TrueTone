import soundfile as sf
import numpy as np

# Write a 16-bit PCM wav
data = np.array([32767, -32768, 0], dtype=np.int16)
sf.write("test.flac", data, 16000)

x, sr = sf.read("test.flac")
print("dtype:", x.dtype)
print("max:", x.max())
print("min:", x.min())
