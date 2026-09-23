import os
import json
import pandas as pd
from datasets import load_dataset
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
from app.preprocessing.pipeline import preprocess_for_detection, prepare_aasist_context
from app.detection.aasist import AasistDetector

def compute_eer(y_true, y_score):
    from sklearn.metrics import roc_curve
    fpr, tpr, thresholds = roc_curve(y_true, y_score)
    fnr = 1 - tpr
    eer_threshold = thresholds[np.nanargmin(np.absolute((fnr - fpr)))]
    eer = fpr[np.nanargmin(np.absolute((fnr - fpr)))]
    return eer

def main():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    os.makedirs(os.path.join(base_dir, "data/splits"), exist_ok=True)
    os.makedirs(os.path.join(base_dir, "reports"), exist_ok=True)
    
    print("Loading test dataset...")
    # Load test set for baseline
    # Test set is 1.5GB, which is manageable to download fully
    ds_test = load_dataset("SherryT997/IndicTTS-Deepfake-Challenge-Data", split="test")
    
    aasist = AasistDetector(os.path.join(base_dir, "third_party/aasist/models/weights/AASIST-L.pth"))
    
    y_true = []
    y_scores = []
    y_pred = []
    
    print(f"Evaluating baseline on {len(ds_test)} samples...")
    for i, ex in enumerate(ds_test):
        is_tts = ex["is_tts"] # 0=real, 1=synthetic
        # Audio is decoded
        audio_array = ex["audio"]["array"]
        sr = ex["audio"]["sampling_rate"]
        
        y_proc = preprocess_for_detection(audio_array, sr)
        
        # Test bench style lookahead (we have whole file)
        # But for overall file evaluation, we just evaluate the first 4.0375s
        # or maybe we should chunk? Let's just evaluate the first 64600 samples
        context = prepare_aasist_context(y_proc[:64600])
        
        res = aasist.predict_aasist(context, 16000)
        spoof_score = res["spoof_score"]
        
        y_true.append(is_tts)
        y_scores.append(spoof_score)
        y_pred.append(1 if spoof_score > 0.5 else 0)
        
        if (i+1) % 100 == 0:
            print(f"Processed {i+1}/{len(ds_test)}")
            
    y_true = np.array(y_true)
    y_scores = np.array(y_scores)
    y_pred = np.array(y_pred)
    
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred)
    rec = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    roc = roc_auc_score(y_true, y_scores)
    eer = compute_eer(y_true, y_scores)
    
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    fpr = fp / (fp + tn)
    fnr = fn / (fn + tp)
    
    report = {
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1": f1,
        "roc_auc": roc,
        "eer": eer,
        "fpr": fpr,
        "fnr": fnr
    }
    
    with open(os.path.join(base_dir, "reports/aasist_pretrained_baseline.json"), "w") as f:
        json.dump(report, f, indent=2)
        
    print("Baseline report saved.")
    
if __name__ == "__main__":
    main()
