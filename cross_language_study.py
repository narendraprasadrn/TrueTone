import json
import os
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import warnings
warnings.filterwarnings('ignore')
import librosa
from feature_extractor_v3 import extract_features_v3
import sys

# To get extract_features_v2
sys.path.append(os.path.abspath('truetone/backend'))
from feature_extractor import extract_features_v2

# Load the V3 cache
cache_file = "data/features/prosody/v3/feature_cache.jsonl"
manifest_dfs = [pd.read_csv(f) for f in ["data/splits_multilingual/internal_train.csv", "data/splits_multilingual/internal_val.csv", "data/splits_multilingual/test.csv"]]
all_manifest = pd.concat(manifest_dfs).set_index('id')

with open("feature_names_v3.json", "r") as fn:
    v3_info = json.load(fn)
v3_names = [r["name"] for r in v3_info]
v3_fams = [r["family"] for r in v3_info]

data = []
with open(cache_file, "r") as f:
    for line in f:
        row = json.loads(line)
        if not row.get("error") and row["id"] in all_manifest.index:
            feats = row["features"]
            vec = [feats.get(n, 0.0) for n in v3_names]
            manifest_row = all_manifest.loc[row["id"]]
            if isinstance(manifest_row, pd.DataFrame):
                manifest_row = manifest_row.iloc[0]
            data.append({
                "id": row["id"],
                "language": manifest_row["language"],
                "is_tts": manifest_row["is_tts"],
                "group": '_'.join(row["id"].split('_')[:2]),
                "features": vec
            })

df = pd.DataFrame(data)
df = df[df["language"].isin(["Tamil", "English", "Hindi"])].copy()
df.reset_index(drop=True, inplace=True)

X = np.array(df["features"].tolist())
y = df["is_tts"].values
langs = df["language"].values

lgb_params = {
    'objective': 'binary',
    'learning_rate': 0.03,
    'num_leaves': 31,
    'max_depth': -1,
    'min_child_samples': 20,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'reg_alpha': 0.1,
    'reg_lambda': 0.1,
    'n_estimators': 1000,
    'verbose': -1
}

def evaluate(y_true, y_pred, threshold=0.5):
    y_pred_bin = (y_pred >= threshold).astype(int)
    auc = roc_auc_score(y_true, y_pred) if len(np.unique(y_true)) > 1 else 0.0
    return {
        "roc_auc": float(auc),
        "accuracy": float(accuracy_score(y_true, y_pred_bin)),
        "precision": float(precision_score(y_true, y_pred_bin, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred_bin, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred_bin, zero_division=0)),
        "cm": confusion_matrix(y_true, y_pred_bin).tolist()
    }

def get_dist(scores):
    if len(scores) == 0:
        return {"mean": 0, "median": 0, "std": 0, "min": 0, "max": 0}
    return {
        "mean": float(np.mean(scores)),
        "median": float(np.median(scores)),
        "std": float(np.std(scores)),
        "min": float(np.min(scores)),
        "max": float(np.max(scores))
    }

experiments = [
    {"name": "Exp A (Ta+Hi -> En)", "train": ["Tamil", "Hindi"], "test": "English", "groups": 2, "type": "Group-holdout (Speaker-Independent Proxy)"},
    {"name": "Exp B (Ta+En -> Hi)", "train": ["Tamil", "English"], "test": "Hindi", "groups": 2, "type": "Group-holdout (Speaker-Independent Proxy)"},
    {"name": "Exp C (En+Hi -> Ta)", "train": ["English", "Hindi"], "test": "Tamil", "groups": 1, "type": "Exploratory cross-language evaluation \u2014 Tamil group limitation"}
]

results = []
score_dists = []
models = {}
feature_importances = []

for exp in experiments:
    train_idx = np.where(np.isin(langs, exp["train"]))[0]
    test_idx = np.where(langs == exp["test"])[0]
    
    X_tr, y_tr = X[train_idx], y[train_idx]
    X_te, y_te = X[test_idx], y[test_idx]
    
    model = lgb.LGBMClassifier(**lgb_params)
    model.fit(X_tr, y_tr)
    
    preds = model.predict_proba(X_te)[:, 1]
    metrics = evaluate(y_te, preds)
    
    results.append({
        "Model": exp["name"],
        "Training Languages": "+".join(exp["train"]),
        "Test Language": exp["test"],
        "Evaluation Type": exp["type"],
        "Groups": exp["groups"],
        "AUC": metrics["roc_auc"],
        "Accuracy": metrics["accuracy"],
        "Precision": metrics["precision"],
        "Recall": metrics["recall"],
        "F1": metrics["f1"]
    })
    
    real_preds = preds[y_te == 0]
    synth_preds = preds[y_te == 1]
    
    real_dist = get_dist(real_preds)
    synth_dist = get_dist(synth_preds)
    
    score_dists.append({
        "Model": exp["name"],
        "Test Language": exp["test"],
        "Real_Mean": real_dist["mean"],
        "Real_Median": real_dist["median"],
        "Real_Std": real_dist["std"],
        "Real_Min": real_dist["min"],
        "Real_Max": real_dist["max"],
        "Synth_Mean": synth_dist["mean"],
        "Synth_Median": synth_dist["median"],
        "Synth_Std": synth_dist["std"],
        "Synth_Min": synth_dist["min"],
        "Synth_Max": synth_dist["max"]
    })
    
    booster = model.booster_
    gain = booster.feature_importance(importance_type='gain')
    models[exp["name"]] = booster
    
    for i, name in enumerate(v3_names):
        feature_importances.append({
            "Model": exp["name"],
            "Feature": name,
            "Family": v3_fams[i],
            "Gain": float(gain[i])
        })

pd.DataFrame(results).to_csv("reports/prosody/v3/cross_language_results.csv", index=False)
with open("reports/prosody/v3/cross_language_summary.json", "w") as f:
    json.dump(results, f, indent=2)

pd.DataFrame(score_dists).to_csv("reports/prosody/v3/score_distributions.csv", index=False)

fi_df = pd.DataFrame(feature_importances)
fam_df = fi_df.groupby(["Model", "Family"])["Gain"].sum().reset_index()
fam_df.to_csv("reports/prosody/v3/feature_family_analysis.csv", index=False)

beh_features = [
    "pause_count", "pause_mean_dur", "pause_median_dur", 
    "speech_segment_count", "speech_silence_ratio", "temporal_regularity", "voiced_ratio"
]
beh_df = fi_df[fi_df["Feature"].isin(beh_features)]
beh_df.to_csv("reports/prosody/v3/behavioural_feature_analysis.csv", index=False)

wa_results = {}
try:
    arr, sr = librosa.load("truetone/backend/real_whatsapp_uncompressed.wav", sr=16000)
    
    try:
        v2_feats = extract_features_v2(arr, sr)
        v2_booster = lgb.Booster(model_file="models/prosody/lightgbm_indictts_v1.txt")
        v2_model_feats = v2_booster.feature_name()
        wa_v2_vec = [v2_feats.get(n, 0.0) for n in v2_model_feats]
        wa_results["V2 (Indictts)"] = float(v2_booster.predict([wa_v2_vec])[0])
    except Exception as e:
        wa_results["V2 (Indictts)"] = f"Failed: {e}"

    v3_feats = extract_features_v3(arr, sr)
    wa_v3_vec = [v3_feats.get(n, 0.0) for n in v3_names]
    
    try:
        comb_booster = lgb.Booster(model_file="models/prosody/v3/lightgbm_teh_combined_v1.txt")
        wa_results["V3 Combined (T+E+H)"] = float(comb_booster.predict([wa_v3_vec])[0])
    except: pass
    try:
        ta_booster = lgb.Booster(model_file="models/prosody/v3/lightgbm_tamil_v1.txt")
        wa_results["V3 Tamil"] = float(ta_booster.predict([wa_v3_vec])[0])
    except: pass
    try:
        en_booster = lgb.Booster(model_file="models/prosody/v3/lightgbm_english_v1.txt")
        wa_results["V3 English"] = float(en_booster.predict([wa_v3_vec])[0])
    except: pass
    try:
        hi_booster = lgb.Booster(model_file="models/prosody/v3/lightgbm_hindi_v1.txt")
        wa_results["V3 Hindi"] = float(hi_booster.predict([wa_v3_vec])[0])
    except: pass
    
    for name, booster in models.items():
        wa_results[name] = float(booster.predict([wa_v3_vec])[0])
        
except Exception as e:
    wa_results["Error"] = str(e)

with open("reports/prosody/v3/whatsapp_cross_model_results.json", "w") as f:
    json.dump(wa_results, f, indent=2)

print("Study complete.")
