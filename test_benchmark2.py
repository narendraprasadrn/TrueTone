import numpy as np
import time
import librosa
from feature_extractor import extract_features_v2
import multiprocessing
import traceback

def worker(arr):
    return extract_features_v2(arr, 16000)

if __name__ == '__main__':
    y, sr = librosa.load('truetone/backend/real_whatsapp_uncompressed.wav', sr=16000)
    samples = [y.copy() for _ in range(20)]
    
    print("Testing OLD (single) extraction on 20 identical samples...")
    t0 = time.time()
    for s in samples:
        worker(s)
    t1 = time.time()
    print(f"OLD: {t1-t0:.2f}s total ({(t1-t0)/20:.2f}s per sample)")
    
    for w in [2, 4, 8, 12]:
        print(f"Testing NEW (parallel {w} workers) on 20 identical samples...")
        t0 = time.time()
        pool = multiprocessing.Pool(w)
        new_res = pool.map(worker, samples)
        pool.close()
        pool.join()
        t1 = time.time()
        print(f"NEW ({w}): {t1-t0:.2f}s total ({(t1-t0)/20:.2f}s per sample)")
