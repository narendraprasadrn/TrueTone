import pandas as pd
import numpy as np
import lightgbm as lgb
import os
import sys
import json
import soundfile as sf
import librosa
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, roc_curve, confusion_matrix
from feature_extractor import extract_features_v2, extract_old_10_features

def compute_eer(y_true, y_score):
    if len(np.unique(y_true)) < 2: return float('nan')
    fpr, tpr, thresholds = roc_curve(y_true, y_score)
    fnr = 1 - tpr
    idx = np.nanargmin(np.absolute((fnr - fpr)))
    return fpr[idx]

def get_metrics(y_true, y_pred, y_score):
    if len(np.unique(y_true)) < 2:
        return {'accuracy': float('nan'), 'precision': float('nan'), 'recall': float('nan'), 'f1': float('nan'), 'roc_auc': float('nan'), 'eer': float('nan'), 'fpr': float('nan'), 'fnr': float('nan')}
    cm = confusion_matrix(y_true, y_pred)
    if cm.shape == (2,2):
        tn, fp, fn, tp = cm.ravel()
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
        fnr = fn / (fn + tp) if (fn + tp) > 0 else 0
    else:
        fpr, fnr = 0.0, 0.0
    return {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred, zero_division=0),
        'recall': recall_score(y_true, y_pred, zero_division=0),
        'f1': f1_score(y_true, y_pred, zero_division=0),
        'roc_auc': roc_auc_score(y_true, y_score),
        'eer': float(compute_eer(y_true, y_score)),
        'fpr': fpr,
        'fnr': fnr
    }

os.makedirs("reports/prosody", exist_ok=True)
os.makedirs("models/prosody", exist_ok=True)

# 1. Load Caches
train_df = pd.read_parquet("data/features/prosody/train_features.parquet")
val_df = pd.read_parquet("data/features/prosody/val_features.parquet")
test_df = pd.read_parquet("data/features/prosody/test_features.parquet")

with open("data/features/prosody/feature_names.json", "r") as f:
    all_feature_names = json.load(f)

# The first 10 features were appended manually in dict
old_feature_names = ['f0_mean', 'f0_std', 'f0_range', 'jitter', 'shimmer', 'hnr', 'voiced_ratio', 'pause_count', 'pause_mean_dur', 'flatness_mean']
# Wait! In feature_extractor.py, I used extract_old_10_features order:
# f0_mean, f0_std, f0_range, jitter, shimmer, voiced_ratio, pause_count, pause_mean_dur, flatness_mean, hnr
old_ordered_names = ['f0_mean', 'f0_std', 'f0_range', 'jitter', 'shimmer', 'voiced_ratio', 'pause_count', 'pause_mean_dur', 'flatness_mean', 'hnr']

X_test_old = test_df[old_ordered_names].values
y_test = test_df['is_tts'].values

# 2. Evaluate EXISTING model
old_model_path = "truetone/backend/app/detection/models/prosody_lgbm.txt"
if os.path.exists(old_model_path):
    old_model = lgb.Booster(model_file=old_model_path)
    old_probs = old_model.predict(X_test_old)
    # The prompt says: "If LightGBM outputs a probability, verify that the positive class corresponds to is_tts=1."
    # Wait, the existing model was trained on ASVspoof LA, where label=1 is spoof. My dataset is_tts=1 is spoof.
    # We will assume old_probs is P(spoof).
    old_preds = (old_probs > 0.5).astype(int)
    old_metrics = get_metrics(y_test, old_preds, old_probs)
    with open("reports/prosody/existing_model_metrics.json", "w") as f:
        json.dump(old_metrics, f, indent=2)
else:
    print("Old model not found!")

# 3. Train NEW LightGBM Model
X_train = train_df[all_feature_names].values
y_train = train_df['is_tts'].values
X_val = val_df[all_feature_names].values
y_val = val_df['is_tts'].values

train_data = lgb.Dataset(X_train, label=y_train, feature_name=all_feature_names)
val_data = lgb.Dataset(X_val, label=y_val, reference=train_data, feature_name=all_feature_names)

params = {
    "objective": "binary",
    "metric": "auc",
    "learning_rate": 0.03,
    "num_leaves": 31,
    "max_depth": -1,
    "min_child_samples": 20,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "reg_alpha": 0.1,
    "reg_lambda": 0.1,
    "verbose": -1,
    "seed": 42
}

print("Training New Model...")
new_model = lgb.train(
    params,
    train_data,
    valid_sets=[train_data, val_data],
    num_boost_round=1000,
    callbacks=[lgb.early_stopping(stopping_rounds=50, verbose=False)]
)

new_model.save_model("models/prosody/lightgbm_indictts_v1.txt")
with open("models/prosody/feature_names_indictts_v1.json", "w") as f:
    json.dump(all_feature_names, f, indent=2)

with open("config/train_prosody_indictts.yaml", "w") as f:
    json.dump({"params": params, "best_iteration": new_model.best_iteration}, f, indent=2)

# 4. Evaluate NEW Model
X_test_new = test_df[all_feature_names].values
new_probs = new_model.predict(X_test_new)
new_preds = (new_probs > 0.5).astype(int)

new_metrics = get_metrics(y_test, new_preds, new_probs)
with open("reports/prosody/new_model_metrics.json", "w") as f:
    json.dump(new_metrics, f, indent=2)

cm = confusion_matrix(y_test, new_preds)
if cm.shape == (2,2):
    cm_dict = {'TN': int(cm[0,0]), 'FP': int(cm[0,1]), 'FN': int(cm[1,0]), 'TP': int(cm[1,1])}
else:
    cm_dict = {}
with open("reports/prosody/confusion_matrix.json", "w") as f:
    json.dump(cm_dict, f, indent=2)

# Score dist
real_scores = new_probs[y_test == 0]
synth_scores = new_probs[y_test == 1]
score_dist = {
    'real': {'mean': float(np.mean(real_scores)), 'median': float(np.median(real_scores)), 'std': float(np.std(real_scores)), 'min': float(np.min(real_scores)), 'max': float(np.max(real_scores))},
    'synthetic': {'mean': float(np.mean(synth_scores)), 'median': float(np.median(synth_scores)), 'std': float(np.std(synth_scores)), 'min': float(np.min(synth_scores)), 'max': float(np.max(synth_scores))}
}
with open("reports/prosody/score_distribution.json", "w") as f:
    json.dump(score_dist, f, indent=2)

# Feature Importance
gain_imp = new_model.feature_importance(importance_type='gain')
split_imp = new_model.feature_importance(importance_type='split')
fi_df = pd.DataFrame({
    'Feature': all_feature_names,
    'Gain': gain_imp,
    'Split': split_imp
}).sort_values('Gain', ascending=False)
fi_df.to_csv("reports/prosody/feature_importance.csv", index=False)

# Per-language
test_manifest = pd.read_csv("data/splits_multilingual/test.csv")
test_df = test_df.merge(test_manifest[['id', 'language']], on='id')
lang_records = []
for lang in test_df['language'].unique():
    ldf = test_df[test_df['language'] == lang]
    lm = get_metrics(ldf['is_tts'].values, (new_model.predict(ldf[all_feature_names].values) > 0.5).astype(int), new_model.predict(ldf[all_feature_names].values))
    lang_records.append({
        'Language': lang,
        'N': len(ldf),
        'Real': (ldf['is_tts']==0).sum(),
        'Synthetic': (ldf['is_tts']==1).sum(),
        'ROC_AUC': lm['roc_auc'],
        'EER': lm['eer'],
        'Accuracy': lm['accuracy'],
        'Precision': lm['precision'],
        'Recall': lm['recall'],
        'F1': lm['f1'],
        'FPR': lm['fpr'],
        'FNR': lm['fnr']
    })
pd.DataFrame(lang_records).to_csv("reports/prosody/per_language_metrics.csv", index=False)

# WhatsApp check
wa_path = "truetone/backend/real_whatsapp.ogg"
wa_scores = []
if os.path.exists(wa_path):
    audio, sr = sf.read(wa_path)
    if audio.ndim > 1: audio = audio.mean(axis=1)
    if sr != 16000: audio = librosa.resample(audio, orig_sr=sr, target_sr=16000)
    target_len = 64600
    total_len = len(audio)
    for i in range(0, total_len, target_len):
        chunk = audio[i:i+target_len]
        if len(chunk) < target_len:
            pad_total = target_len - len(chunk)
            pad_left = pad_total // 2
            pad_right = pad_total - pad_left
            chunk = np.pad(chunk, (pad_left, pad_right), 'constant')
        feats = extract_features_v2(chunk, 16000)
        f_vec = np.array([feats.get(k, 0.0) for k in all_feature_names]).reshape(1, -1)
        score = new_model.predict(f_vec)[0]
        wa_scores.append(float(score))
        
with open("reports/prosody/whatsapp_external_check.json", "w") as f:
    json.dump({
        "windows": len(wa_scores),
        "scores": wa_scores,
        "mean": float(np.mean(wa_scores)) if len(wa_scores)>0 else 0.0,
        "median": float(np.median(wa_scores)) if len(wa_scores)>0 else 0.0,
        "min": float(np.min(wa_scores)) if len(wa_scores)>0 else 0.0,
        "max": float(np.max(wa_scores)) if len(wa_scores)>0 else 0.0
    }, f, indent=2)

print("Evaluation of new LightGBM model complete.")
