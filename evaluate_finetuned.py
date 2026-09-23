import pandas as pd
import numpy as np
import os
import glob
import sys
import json
import torch
import soundfile as sf
import librosa
from datasets import load_dataset, Audio
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, roc_curve, confusion_matrix
import warnings
import hashlib
warnings.filterwarnings("ignore")

sys.path.append(os.path.abspath("truetone/backend"))
from app.detection.aasist import AasistDetector

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
            'accuracy': accuracy_score(y_true, y_pred) if len(y_true)>0 else float('nan'),
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
        waveform = waveform.mean(axis=0)
    waveform = waveform.flatten()
    if sr != 16000:
        waveform = librosa.resample(waveform, orig_sr=sr, target_sr=16000)
    target_len = 64600
    curr_len = len(waveform)
    if curr_len > target_len:
        start = (curr_len - target_len) // 2
        waveform = waveform[start:start+target_len]
    elif curr_len < target_len:
        pad_total = target_len - curr_len
        pad_left = pad_total // 2
        pad_right = pad_total - pad_left
        waveform = np.pad(waveform, (pad_left, pad_right), 'constant')
    return waveform

os.makedirs("reports/finetuning_tamil_hindi", exist_ok=True)

print("Loading Fine-Tuned Model...")
ckpt_path = os.path.abspath("models/aasist/finetuned_tamil_hindi/best.pt")
detector = AasistDetector(ckpt_path, device="cuda" if torch.cuda.is_available() else "cpu")

print("Evaluating on Multilingual Test Set...")
test_df = pd.read_csv("data/splits_multilingual/test.csv")
test_ids = set(test_df['id'].tolist())
test_dict = test_df.set_index('id').to_dict('index')

cache_dir = os.path.expanduser("~/.cache/huggingface/hub/datasets--SherryT997--IndicTTS-Deepfake-Challenge-Data/snapshots/*/data/train-*.parquet")
all_files = glob.glob(cache_dir)
expected_shards = [0, 1, 3, 4, 6, 7, 9, 10, 12, 13, 15, 16, 18, 19, 21, 22, 24, 25, 27, 28, 30, 31, 33]
files_to_load = [f for f in all_files if int(os.path.basename(f).split('-')[1]) in expected_shards]

ds = load_dataset("parquet", data_files=files_to_load, split="train")
ds = ds.cast_column("audio", Audio(sampling_rate=16000))

predictions = []

for item in ds:
    idx = item['id']
    if idx in test_ids:
        try:
            audio_array = item['audio']['array']
            sr = item['audio']['sampling_rate']
            window = process_audio(audio_array, sr)
            res = detector.predict_aasist(window, sr=16000)
            
            is_tts_actual = test_dict[idx]['is_tts']
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
            print(f"Failed {idx}: {e}")

pred_df = pd.DataFrame(predictions)
pred_df.to_csv("reports/finetuning_tamil_hindi/fine_tuned_predictions.csv", index=False)

y_true = pred_df['is_tts']
y_pred = pred_df['predicted_label']
y_score = pred_df['spoof_score']

multi_metrics = get_metrics(y_true, y_pred, y_score)
with open("reports/finetuning_tamil_hindi/multilingual_metrics.json", "w") as f:
    json.dump(multi_metrics, f, indent=2)

cm = confusion_matrix(y_true, y_pred)
if len(np.unique(y_true)) > 1:
    cm_dict = {'TN': int(cm[0,0]), 'FP': int(cm[0,1]), 'FN': int(cm[1,0]), 'TP': int(cm[1,1])}
else:
    cm_dict = {}
with open("reports/finetuning_tamil_hindi/confusion_matrix_multilingual.json", "w") as f:
    json.dump(cm_dict, f, indent=2)
    
lang_results = []
for lang in pred_df['language'].unique():
    ldf = pred_df[pred_df['language'] == lang]
    lm = get_metrics(ldf['is_tts'], ldf['predicted_label'], ldf['spoof_score'])
    lang_results.append({
        'Language': lang,
        'ROC_AUC': lm['roc_auc'],
        'EER': lm['eer'],
        'F1': lm['f1'],
        'Recall': lm['recall'],
        'Accuracy': lm['accuracy']
    })
lang_df = pd.DataFrame(lang_results)
lang_df.to_csv("reports/finetuning_tamil_hindi/per_language_finetuned.csv", index=False)

# WhatsApp check
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
        
with open("reports/finetuning_tamil_hindi/whatsapp_external_check.json", "w") as f:
    json.dump({"scores": [float(x) for x in wa_scores]}, f)

# Hindi metrics (extract from lang_df)
hindi_metrics = lang_df[lang_df['Language'] == 'Hindi'].to_dict('records')
if hindi_metrics:
    with open("reports/finetuning_tamil_hindi/hindi_metrics.json", "w") as f:
        json.dump(hindi_metrics[0], f, indent=2)

# Tamil metrics: Tamil isn't in test set!
with open("reports/finetuning_tamil_hindi/tamil_metrics.json", "w") as f:
    json.dump({"ROC_AUC": "N/A", "note": "Tamil test set is empty due to group isolation limitations."}, f, indent=2)

print("Evaluation of fine-tuned model complete.")

# Create the comparison CSV
base_lang_df = pd.read_csv("reports/baseline/aasist_l_per_language.csv")
comp_records = []
for lang in set(base_lang_df['Language']).union(set(lang_df['Language'])):
    br = base_lang_df[base_lang_df['Language'] == lang]
    fr = lang_df[lang_df['Language'] == lang]
    
    b_auc = br['ROC_AUC'].iloc[0] if not br.empty else float('nan')
    f_auc = fr['ROC_AUC'].iloc[0] if not fr.empty else float('nan')
    b_eer = br['EER'].iloc[0] if not br.empty else float('nan')
    f_eer = fr['EER'].iloc[0] if not fr.empty else float('nan')
    
    comp_records.append({
        'Language': lang,
        'Baseline_AUC': b_auc,
        'Finetuned_AUC': f_auc,
        'Delta_AUC': f_auc - b_auc,
        'Baseline_EER': b_eer,
        'Finetuned_EER': f_eer,
        'Delta_EER': f_eer - b_eer
    })
    
pd.DataFrame(comp_records).to_csv("reports/finetuning_tamil_hindi/pretrained_vs_finetuned.csv", index=False)
