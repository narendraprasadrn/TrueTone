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
    
    print("Testing OLD (single) extraction on 10 identical samples...")
    samples = [y.copy() for _ in range(10)]
    t0 = time.time()
    old_res = []
    for s in samples:
        old_res.append(worker(s))
    t1 = time.time()
    old_time = t1 - t0
    print(f"OLD: {old_time:.2f}s total ({old_time/10:.2f}s per sample)")
    
    print("Testing NEW (parallel 4 workers) on 10 identical samples...")
    t0 = time.time()
    pool = multiprocessing.Pool(4)
    new_res = pool.map(worker, samples)
    pool.close()
    pool.join()
    t1 = time.time()
    new_time = t1 - t0
    print(f"NEW: {new_time:.2f}s total ({new_time/10:.2f}s per sample)")
    
    # Check determinism
    mismatches = 0
    max_diff = 0.0
    for i in range(10):
        for k in old_res[i].keys():
            diff = abs(old_res[i][k] - new_res[i][k])
            if diff > 1e-6:
                print(f"Mismatch in {k}: {old_res[i][k]} != {new_res[i][k]}")
                mismatches += 1
            if diff > max_diff: max_diff = diff
            
    print(f"\nDeterminism check:")
    print(f"Max absolute difference: {max_diff}")
    if mismatches == 0:
        print("Features are perfectly equivalent!")
        
