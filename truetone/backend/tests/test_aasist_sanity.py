import os
import sys
import numpy as np
import soundfile as sf
import json

base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, base_dir)

from app.detection.aasist import AasistDetector
from app.preprocessing.pipeline import preprocess_for_detection

def run_sanity_test():
    checkpoint_path = os.path.join(base_dir, "third_party/aasist/models/weights/AASIST-L.pth")
    detector = AasistDetector(checkpoint_path)
    
    # Files
    bonafide_file = os.path.join(base_dir, "real_whatsapp_uncompressed.wav")
    spoof_file = os.path.join(base_dir, "synthetic_cloned.ogg")
    
    for filepath in [bonafide_file, spoof_file]:
        print(f"\n--- Testing: {os.path.basename(filepath)} ---")
        if not os.path.exists(filepath):
            print("File not found.")
            continue
            
        audio, sr = sf.read(filepath)
        duration = len(audio) / sr
        
        # Take 4.0375s
        window = audio[:int(4.0375 * sr)]
        
        processed = preprocess_for_detection(window, sr, debug=False)
        
        # We can directly pad here or let predict_aasist do it
        res = detector.predict_aasist(processed, 16000)
        
        print(f"file: {os.path.basename(filepath)}")
        print(f"duration: {duration:.4f}s")
        print(f"sample_rate: {sr}")
        print(f"processed_samples: {len(processed)}")
        print(f"logit_0: {res['logits'][0]:.4f}")
        print(f"logit_1: {res['logits'][1]:.4f}")
        print(f"spoof_score: {res['spoof_score']:.4f}")
        print(f"bonafide_score: {res['bonafide_score']:.4f}")
        print(f"predicted_class: {res['predicted_class']}")

if __name__ == "__main__":
    run_sanity_test()
