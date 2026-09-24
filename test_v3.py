import numpy as np
import librosa

def test_delta(y, sr):
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    n_frames = mfcc.shape[1]
    width = min(9, n_frames if n_frames % 2 != 0 else n_frames - 1)
    if width < 3:
        delta = np.zeros_like(mfcc)
        delta2 = np.zeros_like(mfcc)
    else:
        delta = librosa.feature.delta(mfcc, width=width)
        delta2 = librosa.feature.delta(mfcc, order=2, width=width)
    return delta.shape, delta2.shape

# Very short audio
y = np.random.randn(2000) 
print(test_delta(y, 16000))
