import librosa
import soundfile as sf
import numpy as np

audio_data, sr = librosa.load(librosa.ex('libri1'), sr=None)
audio_16k = librosa.resample(audio_data, orig_sr=sr, target_sr=16000)

window4 = audio_16k[64000:64000+64600]
sf.write("window4.wav", window4, 16000)
