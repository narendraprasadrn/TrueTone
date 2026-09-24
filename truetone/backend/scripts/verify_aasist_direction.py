import os
import sys
import glob
import numpy as np
import librosa
from sklearn.metrics import roc_auc_score, roc_curve

# Add the parent directory to sys.path so we can import from app
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.preprocessing.pipeline import preprocess_for_detection, prepare_aasist_context
from app.detection.aasist import AasistDetector

def compute_eer(y_true, y_score):
    fpr, tpr, thresholds = roc_curve(y_true, y_score, pos_label=1)
    fnr = 1 - tpr
    eer_threshold = thresholds[np.nanargmin(np.absolute((fnr - fpr)))]
    eer = fpr[np.nanargmin(np.absolute((fnr - fpr)))]
    return eer

def run_verification():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    real_dir = os.path.join(base_dir, "data", "verify", "real")
    fake_dir = os.path.join(base_dir, "data", "verify", "fake")
    
    detector = AasistDetector(os.path.join(base_dir, "third_party/aasist/models/weights/AASIST-L.pth"))
    
    real_files = glob.glob(os.path.join(real_dir, "*.*"))
    fake_files = glob.glob(os.path.join(fake_dir, "*.*"))
    
    if not real_files and not fake_files:
        print("No files found in data/verify/real or data/verify/fake. Waiting for data.")
        return
        
    scores = []
    labels = []
    
    print("--- REAL CLIPS ---")
    real_scores = []
    for file in real_files:
        audio, sr = librosa.load(file, sr=None)
        audio_processed = preprocess_for_detection(audio, sr)
        aasist_ctx = prepare_aasist_context(audio_processed)
        res = detector.predict_aasist(aasist_ctx, 16000)
        spoof_score = res["spoof_score"]
        real_scores.append(spoof_score)
        scores.append(spoof_score)
        labels.append(0) # Real = 0 (Fake is positive class for AUC)
        print(f"File: {os.path.basename(file)} | Logits: {res['logits']} | Softmax (Spoof, Bonafide): {res['spoof_score']:.4f}, {res['bonafide_score']:.4f} | Final Spoof Score: {spoof_score:.4f}")
        
    print("\n--- FAKE CLIPS ---")
    fake_scores = []
    for file in fake_files:
        audio, sr = librosa.load(file, sr=None)
        audio_processed = preprocess_for_detection(audio, sr)
        aasist_ctx = prepare_aasist_context(audio_processed)
        res = detector.predict_aasist(aasist_ctx, 16000)
        spoof_score = res["spoof_score"]
        fake_scores.append(spoof_score)
        scores.append(spoof_score)
        labels.append(1) # Fake = 1
        print(f"File: {os.path.basename(file)} | Logits: {res['logits']} | Softmax (Spoof, Bonafide): {res['spoof_score']:.4f}, {res['bonafide_score']:.4f} | Final Spoof Score: {spoof_score:.4f}")
        
    print("\n--- AGGREGATE RESULTS ---")
    if real_scores:
        print(f"REAL - Mean Spoof Score: {np.mean(real_scores):.4f}, Std: {np.std(real_scores):.4f}")
    if fake_scores:
        print(f"FAKE - Mean Spoof Score: {np.mean(fake_scores):.4f}, Std: {np.std(fake_scores):.4f}")
        
    if real_scores and fake_scores:
        auc = roc_auc_score(labels, scores)
        eer = compute_eer(labels, scores)
        print(f"AUC: {auc:.4f}")
        print(f"EER: {eer:.4f}")

if __name__ == "__main__":
    run_verification()
