import pandas as pd
import numpy as np
import os
import glob
from datasets import load_dataset, Audio
import sys
import json
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, roc_curve, confusion_matrix
import warnings
warnings.filterwarnings("ignore")

# Force using backend imports
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

def process_audio(waveform, sr):
    if waveform.ndim > 1 and waveform.shape[0] > 1:
        waveform = waveform.mean(axis=0) # mono
    waveform = waveform.flatten()
    
    if sr != 16000:
        waveform = librosa.resample(waveform, orig_sr=sr, target_sr=16000)
    
    target_len = 64600
    curr_len = len(waveform)
    
    if curr_len > target_len:
        # Center crop
        start = (curr_len - target_len) // 2
        waveform = waveform[start:start+target_len]
    elif curr_len < target_len:
        # Zero-pad (center pad)
        pad_total = target_len - curr_len
        pad_left = pad_total // 2
        pad_right = pad_total - pad_left
        waveform = np.pad(waveform, (pad_left, pad_right), 'constant')
    return waveform

def main():
    print("1. MODEL")
    ckpt_path = os.path.abspath("truetone/backend/third_party/aasist/models/weights/AASIST-L.pth")
    print(f"Checkpoint path: {ckpt_path}")
    detector = AasistDetector(ckpt_path, device="cuda" if torch.cuda.is_available() else "cpu")
    
    print("2. AUDIO PREPROCESSING & 4. RUN ON TEST SET")
    test_df = pd.read_csv("data/splits_multilingual/test.csv")
    test_ids = set(test_df['id'].tolist())
    
    # Build dictionary from df for fast lookup
    test_dict = test_df.set_index('id').to_dict('index')
    
    cache_dir = os.path.expanduser("~/.cache/huggingface/hub/datasets--SherryT997--IndicTTS-Deepfake-Challenge-Data/snapshots/*/data/train-*.parquet")
    all_files = glob.glob(cache_dir)
    expected_shards = [0, 1, 3, 4, 6, 7, 9, 10, 12, 13, 15, 16, 18, 19, 21, 22, 24, 25, 27, 28, 30, 31, 33]
    files_to_load = [f for f in all_files if int(os.path.basename(f).split('-')[1]) in expected_shards]
    
    ds = load_dataset("parquet", data_files=files_to_load, split="train")
    ds = ds.cast_column("audio", Audio(sampling_rate=16000))
    
    predictions = []
    missing_ids = set(test_ids)
    failed = 0
    
    # Filter dataset lazily
    for item in ds:
        idx = item['id']
        if idx in test_ids:
            missing_ids.remove(idx)
            try:
                audio_array = item['audio']['array']
                sr = item['audio']['sampling_rate']
                
                # Preprocess
                window = process_audio(audio_array, sr)
                
                # Predict
                res = detector.predict_aasist(window, sr=16000)
                
                # In dataset: 0 = real/bonafide, 1 = synth/spoof
                # In AASIST output: predicted_class = "spoof" or "bonafide"
                # spoof_score is probability of spoof
                is_tts_actual = test_dict[idx]['is_tts']
                
                # We use 0.5 as default threshold for spoof_score
                predicted_label = 1 if res['spoof_score'] > 0.5 else 0
                
                predictions.append({
                    'id': idx,
                    'language': test_dict[idx]['language'],
                    'is_tts': is_tts_actual,
                    'spoof_score': res['spoof_score'],
                    'predicted_label': predicted_label,
                    'threshold_used': 0.5
                })
                
            except Exception as e:
                print(f"Failed to process {idx}: {e}")
                failed += 1

    print(f"Total processed: {len(predictions)}")
    print(f"Missing from dataset: {len(missing_ids)}")
    print(f"Failed processing: {failed}")
    
    pred_df = pd.DataFrame(predictions)
    os.makedirs("reports/baseline", exist_ok=True)
    pred_df.to_csv("reports/baseline/aasist_l_test_predictions.csv", index=False)
    
    print("5. PRIMARY METRICS & 6. PER-LANGUAGE RESULTS")
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
    
    cm_dict = {
        'TN': int(cm[0,0]) if cm.shape == (2,2) else int(cm[0,0]),
        'FP': int(cm[0,1]) if cm.shape == (2,2) else 0,
        'FN': int(cm[1,0]) if cm.shape == (2,2) else 0,
        'TP': int(cm[1,1]) if cm.shape == (2,2) else 0,
    }
    
    with open("reports/baseline/aasist_l_confusion_matrix.json", "w") as f:
        json.dump(cm_dict, f, indent=2)
        
    lang_results = []
    for lang in pred_df['language'].unique():
        ldf = pred_df[pred_df['language'] == lang]
        lm = get_metrics(ldf['is_tts'], ldf['predicted_label'], ldf['spoof_score'])
        lang_results.append({
            'Language': lang,
            'N': len(ldf),
            'Real': int((ldf['is_tts']==0).sum()),
            'Synthetic': int((ldf['is_tts']==1).sum()),
            'Accuracy': lm['accuracy'],
            'Precision': lm['precision'],
            'Recall': lm['recall'],
            'F1': lm['f1'],
            'ROC_AUC': lm['roc_auc'],
            'EER': lm['eer']
        })
        
    lang_df = pd.DataFrame(lang_results)
    lang_df.to_csv("reports/baseline/aasist_l_per_language.csv", index=False)
    
    print("7. SCORE DISTRIBUTION")
    real_scores = pred_df[pred_df['is_tts'] == 0]['spoof_score']
    synth_scores = pred_df[pred_df['is_tts'] == 1]['spoof_score']
    
    score_dist = {
        'real': {
            'mean': float(real_scores.mean()),
            'median': float(real_scores.median()),
            'std': float(real_scores.std()),
            'min': float(real_scores.min()),
            'max': float(real_scores.max()),
            'count': len(real_scores)
        },
        'synthetic': {
            'mean': float(synth_scores.mean()),
            'median': float(synth_scores.median()),
            'std': float(synth_scores.std()),
            'min': float(synth_scores.min()),
            'max': float(synth_scores.max()),
            'count': len(synth_scores)
        }
    }
    with open("reports/baseline/aasist_l_score_distribution.json", "w") as f:
        json.dump(score_dist, f, indent=2)
        
    # Summary JSON
    import datetime
    with open("data/splits_multilingual/test.csv", "rb") as f:
        split_hash = hashlib.sha256(f.read()).hexdigest()
        
    summary = {
        'timestamp': datetime.datetime.now().isoformat(),
        'checkpoint': ckpt_path,
        'preprocessing': 'Mono, 16kHz, EXACTLY 64600 samples (center crop / center pad)',
        'threshold': 0.5,
        'split_manifest_sha256': split_hash,
        'processed_samples': len(predictions),
        'missing_samples': len(missing_ids),
        'failed_samples': failed,
        'overall_metrics': overall_metrics
    }
    with open("reports/baseline/aasist_l_baseline_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
        
    print("8. REAL-WORLD WHATSAPP SAMPLE")
    wa_path = "truetone/backend/real_whatsapp.ogg"
    if os.path.exists(wa_path):
        audio, sr = sf.read(wa_path)
        if audio.ndim > 1: audio = audio.mean(axis=1)
        if sr != 16000:
            audio = librosa.resample(audio, orig_sr=sr, target_sr=16000)
            
        target_len = 64600
        total_len = len(audio)
        
        wa_scores = []
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
    
    print("\nPRETRAINED AASIST-L BASELINE COMPLETE — AWAITING REVIEW BEFORE FINE-TUNING.")

import torch
if __name__ == "__main__":
    main()
