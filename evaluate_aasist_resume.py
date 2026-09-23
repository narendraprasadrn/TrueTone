import pandas as pd
import numpy as np
import os
import sys
import json
import hashlib
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, roc_curve, confusion_matrix
import warnings
warnings.filterwarnings("ignore")

sys.path.append(os.path.abspath("truetone/backend"))
from app.detection.aasist import AasistDetector
import soundfile as sf
import librosa

def compute_eer(y_true, y_score):
    if len(np.unique(y_true)) < 2:
        return float('nan')
    fpr, tpr, thresholds = roc_curve(y_true, y_score)
    fnr = 1 - tpr
    idx = np.nanargmin(np.absolute((fnr - fpr)))
    return fpr[idx]

def get_metrics(y_true, y_pred, y_score):
    if len(np.unique(y_true)) < 2:
        return {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': float('nan'),
            'recall': float('nan'),
            'f1': float('nan'),
            'roc_auc': float('nan'),
            'eer': float('nan')
        }
    return {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred, zero_division=0),
        'recall': recall_score(y_true, y_pred, zero_division=0),
        'f1': f1_score(y_true, y_pred, zero_division=0),
        'roc_auc': roc_auc_score(y_true, y_score),
        'eer': float(compute_eer(y_true, y_score))
    }

print("RESUMING BASELINE SCRIPT")
ckpt_path = os.path.abspath("truetone/backend/third_party/aasist/models/weights/AASIST-L.pth")
pred_df = pd.read_csv("reports/baseline/aasist_l_test_predictions.csv")

y_true = pred_df['is_tts']
y_pred = pred_df['predicted_label']
y_score = pred_df['spoof_score']

overall_metrics = get_metrics(y_true, y_pred, y_score)
cm = confusion_matrix(y_true, y_pred)
if len(np.unique(y_true)) > 1:
    tn, fp, fn, tp = cm.ravel()
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
    fnr = fn / (fn + tp) if (fn + tp) > 0 else 0
else:
    fpr, fnr = 0, 0

overall_metrics['fpr'] = fpr
overall_metrics['fnr'] = fnr

import datetime
with open("data/splits_multilingual/test.csv", "rb") as f:
    split_hash = hashlib.sha256(f.read()).hexdigest()
    
summary = {
    'timestamp': datetime.datetime.now().isoformat(),
    'checkpoint': ckpt_path,
    'preprocessing': 'Mono, 16kHz, EXACTLY 64600 samples (center crop / center pad)',
    'threshold': 0.5,
    'split_manifest_sha256': split_hash,
    'processed_samples': len(pred_df),
    'missing_samples': 0,
    'failed_samples': 0,
    'overall_metrics': overall_metrics
}
with open("reports/baseline/aasist_l_baseline_summary.json", "w") as f:
    json.dump(summary, f, indent=2)

print("8. REAL-WORLD WHATSAPP SAMPLE")
import torch
detector = AasistDetector(ckpt_path, device="cuda" if torch.cuda.is_available() else "cpu")
wa_path = "truetone/backend/real_whatsapp.ogg"
wa_scores = []
if os.path.exists(wa_path):
    audio, sr = sf.read(wa_path)
    if audio.ndim > 1: audio = audio.mean(axis=1)
    if sr != 16000:
        audio = librosa.resample(audio, orig_sr=sr, target_sr=16000)
        
    target_len = 64600
    total_len = len(audio)
    
    for i in range(0, total_len, target_len):
        chunk = audio[i:i+target_len]
        if len(chunk) < target_len:
            pad_total = target_len - len(chunk)
            pad_left = pad_total // 2
            pad_right = pad_total - pad_left
            chunk = np.pad(chunk, (pad_left, pad_right), 'constant')
            
        res = detector.predict_aasist(chunk, sr=16000)
        wa_scores.append(res['spoof_score'])
        
    print(f"\nExternal real-world domain-shift check — NOT part of the benchmark.")
    print(f"WhatsApp file: {wa_path}")
    print(f"Number of 64600-sample windows: {len(wa_scores)}")
    print(f"Scores per window: {[round(s,4) for s in wa_scores]}")
    print(f"Aggregate Mean Score: {np.mean(wa_scores):.4f}")
    print(f"Aggregate Median Score: {np.median(wa_scores):.4f}")

with open("reports/baseline/whatsapp_external_check.json", "w") as f:
    json.dump({"scores": [float(x) for x in wa_scores]}, f)

print("\nPRETRAINED AASIST-L BASELINE COMPLETE — AWAITING REVIEW BEFORE FINE-TUNING.")

