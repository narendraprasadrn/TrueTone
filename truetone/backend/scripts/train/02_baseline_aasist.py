import os
import json
import numpy as np
from datasets import load_dataset
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
from app.preprocessing.pipeline import preprocess_for_detection, prepare_aasist_context
from app.detection.aasist import AasistDetector

def compute_eer(y_true, y_score):
    from sklearn.metrics import roc_curve
    fpr, tpr, thresholds = roc_curve(y_true, y_score)
    fnr = 1 - tpr
    eer = fpr[np.nanargmin(np.absolute((fnr - fpr)))]
    return eer

base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
aasist = AasistDetector(os.path.join(base_dir, "third_party/aasist/models/weights/AASIST-L.pth"))

print("Loading test split (streaming)...")
ds = load_dataset("SherryT997/IndicTTS-Deepfake-Challenge-Data", "default", split="test", streaming=True)

y_true = []
y_scores = []

print("\n--- SMOKE TEST ---")
for i, ex in enumerate(ds):
    label = ex["is_tts"] # 0 = bonafide, 1 = spoof
    lang = ex["language"]
    audio = ex["audio"]["array"]
    sr = ex["audio"]["sampling_rate"]
    
    y_proc = preprocess_for_detection(audio, sr)
    context = prepare_aasist_context(y_proc[:64600])
    
    res = aasist.predict_aasist(context, 16000)
    spoof = res["spoof_score"]
    bonafide = res["bonafide_score"]
    
    if i < 5:
        print(f"File {ex['id']} | Lang: {lang} | Label (is_tts): {label}")
        print(f"  Duration: {len(audio)/sr:.2f}s | Sample Rate: {sr}")
        print(f"  Input Samples: {len(context)}")
        print(f"  Logits: {res['logits']}")
        print(f"  Spoof Score: {spoof:.4f} | Bonafide: {bonafide:.4f}")
        print(f"  Prediction: {'Spoof' if spoof > bonafide else 'Bonafide'}")
        
    y_true.append(label)
    y_scores.append(spoof)
    
    if i == 49: # Just run 50 samples for the baseline estimate
        break

print("\n--- BASELINE METRICS ---")
y_true = np.array(y_true)
y_scores = np.array(y_scores)
y_pred = (y_scores > 0.5).astype(int)

acc = accuracy_score(y_true, y_pred)
prec = precision_score(y_true, y_pred, zero_division=0)
rec = recall_score(y_true, y_pred, zero_division=0)
f1 = f1_score(y_true, y_pred, zero_division=0)
roc = roc_auc_score(y_true, y_scores)
eer = compute_eer(y_true, y_scores)

tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0,1]).ravel()
fpr = fp / (fp + tn) if (fp+tn) > 0 else 0
fnr = fn / (fn + tp) if (fn+tp) > 0 else 0

print(f"Samples: {len(y_true)}")
print(f"Accuracy: {acc:.4f}")
print(f"Precision: {prec:.4f}")
print(f"Recall: {rec:.4f}")
print(f"F1: {f1:.4f}")
print(f"ROC-AUC: {roc:.4f}")
print(f"EER: {eer:.4f}")
print(f"FPR: {fpr:.4f}")
print(f"FNR: {fnr:.4f}")
