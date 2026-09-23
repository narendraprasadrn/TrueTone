import os
import sys
import numpy as np
import librosa
import torch
import torch.nn.functional as F

base_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(base_dir, "truetone/backend"))

from app.detection.aasist import AasistDetector
from app.detection.prosody import ProsodyDetector
from app.preprocessing.pipeline import preprocess_for_detection

def run_diagnostic():
    # 1. Load models
    aasist = AasistDetector(os.path.join(base_dir, "truetone/backend/third_party/aasist/models/weights/AASIST-L.pth"))
    prosody = ProsodyDetector(os.path.join(base_dir, "truetone/backend/app/detection/models/prosody_lgbm.txt"))
    
    # 2. Get audio files
    test_files = [
        ("Real human speech 1", os.path.join(base_dir, "truetone/test_libri.wav")), 
        ("Real human speech 2", os.path.join(base_dir, "truetone/backend/real_whatsapp.ogg")),
        ("Synthetic speech", os.path.join(base_dir, "truetone/backend/synthetic.wav"))
    ]
    
    for label, path in test_files:
        print(f"\n==============================================")
        print(f"FILE: {label} - {path}")
        if not os.path.exists(path):
            print("FILE NOT FOUND")
            continue
            
        audio, sr = librosa.load(path, sr=None)
        if len(audio.shape) > 1:
            audio = audio[0]
            
        print(f"original_sample_rate: {sr}")
        print(f"original_duration: {len(audio)/sr:.4f}")
        
        # We need a 4.0375s window for AASIST conceptually
        max_len_samples = 64600
        proc_audio = preprocess_for_detection(audio, sr, debug=False)
        # AASIST takes processed audio, resampled to 16000
        
        # Slice to exactly 64600 samples for AASIST if possible, or let AASIST pad
        aasist_input = proc_audio[:max_len_samples]
        
        print(f"processed_sample_rate: 16000")
        print(f"waveform_min: {aasist_input.min():.4f}")
        print(f"waveform_max: {aasist_input.max():.4f}")
        
        # 3. AASIST DIAGNOSTIC
        print(f"\n[AASIST - TILED 2s]")
        aasist_input_2s = proc_audio[:32000]
        res_2s = aasist.predict_aasist(aasist_input_2s, 16000)
        print(f"SPOOF SCORE: {res_2s['spoof_score']:.6f}")
        
        print(f"\n[AASIST - CONTIGUOUS 4s]")
        aasist_input_4s = proc_audio[:64600]
        res_4s = aasist.predict_aasist(aasist_input_4s, 16000)
        print(f"SPOOF SCORE: {res_4s['spoof_score']:.6f}")
        
        # 4. PROSODY DIAGNOSTIC
        prosody_input = proc_audio[:32000]
        
        from app.detection.prosody_features import extract_prosody_features
        features = extract_prosody_features(prosody_input, 16000)
        
        print("\n[PROSODY FEATURES]")
        feature_names = ['f0_mean', 'f0_std', 'f0_range', 'jitter', 'shimmer', 'voiced_ratio', 'pause_count', 'pause_mean_dur', 'flatness_mean', 'hnr']
        for name, val in zip(feature_names, features):
            status = "OK"
            if np.isnan(val): status = "NaN"
            elif np.isinf(val): status = "Inf"
            elif val == 0: status = "ZERO"
            print(f"{name}: {val:.6f} [{status}]")
            
        p_score = prosody.score(prosody_input, 16000)
        
        print(f"\n[PROSODY RESULT]")
        print(f"SPOOF SCORE: {p_score:.6f}")

if __name__ == "__main__":
    run_diagnostic()
