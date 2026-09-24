import json
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import GroupKFold, StratifiedKFold
from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import os
import glob
from feature_extractor_v3 import extract_features_v3
import soundfile as sf
import warnings
warnings.filterwarnings('ignore')

# 1. Verify V3 Cache
cache_file = "data/features/prosody/v3/feature_cache.jsonl"
manifest_dfs = [pd.read_csv(f) for f in ["data/splits_multilingual/internal_train.csv", "data/splits_multilingual/internal_val.csv", "data/splits_multilingual/test.csv"]]
all_manifest = pd.concat(manifest_dfs).set_index('id')

data = []
with open(cache_file, "r") as f:
    for line in f:
        row = json.loads(line)
        if not row.get("error") and row["id"] in all_manifest.index:
            feats = row["features"]
            # Enforce deterministic order from feature_names_v3.json
            with open("feature_names_v3.json", "r") as fn:
                v3_names = [r["name"] for r in json.load(fn)]
            
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
X_all = np.array(df["features"].tolist())
y_all = df["is_tts"].values

print("--- V3 CACHE VERIFICATION ---")
print(f"X.shape: {X_all.shape}")
print(f"y.shape: {y_all.shape}")
print(f"Feature dimension: {len(v3_names)}")
print(f"NaN count: {np.isnan(X_all).sum()}")
print(f"Inf count: {np.isinf(X_all).sum()}")

# Ensure labels are 0/1
assert set(y_all).issubset({0, 1}), "Labels are not 0 and 1"

# 2. Build T/E/H Dataset
teh_df = df[df["language"].isin(["Tamil", "English", "Hindi"])].copy()
teh_df.reset_index(drop=True, inplace=True)
print("\n--- T/E/H DATASET COUNTS ---")
for lang in ["Tamil", "English", "Hindi"]:
    ldf = teh_df[teh_df["language"] == lang]
    print(f"{lang}: {len(ldf)} total, {sum(ldf['is_tts']==0)} real, {sum(ldf['is_tts']==1)} synth, {ldf['group'].nunique()} groups")

X_teh = np.array(teh_df["features"].tolist())
y_teh = teh_df["is_tts"].values
groups_teh = teh_df["group"].values
langs_teh = teh_df["language"].values

# LightGBM Params
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
    if len(np.unique(y_true)) > 1:
        auc = roc_auc_score(y_true, y_pred)
    else:
        auc = 0.0
    return {
        "roc_auc": auc,
        "accuracy": accuracy_score(y_true, y_pred_bin),
        "precision": precision_score(y_true, y_pred_bin, zero_division=0),
        "recall": recall_score(y_true, y_pred_bin, zero_division=0),
        "f1": f1_score(y_true, y_pred_bin, zero_division=0),
    }

def train_eval_model(X, y, groups, model_path, use_group_kfold=True, is_exploratory=False):
    # If 1 group, StratifiedKFold (exploratory)
    # If 2 groups, GroupKFold (leave one group out)
    if not use_group_kfold or len(np.unique(groups)) < 2:
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        splits = list(cv.split(X, y))
        mode = "Within-group exploratory evaluation — not speaker-independent."
    else:
        cv = GroupKFold(n_splits=min(5, len(np.unique(groups))))
        splits = list(cv.split(X, y, groups))
        mode = "Group-holdout evaluation (speaker-independent proxy)."

    metrics = []
    models = []
    oof_preds = np.zeros(len(y))
    
    for train_idx, test_idx in splits:
        X_tr, y_tr = X[train_idx], y[train_idx]
        X_val, y_val = X[test_idx], y[test_idx]
        
        # Need evaluation sets for early stopping
        model = lgb.LGBMClassifier(**lgb_params)
        model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], callbacks=[lgb.early_stopping(50, verbose=False)])
        
        preds = model.predict_proba(X_val)[:, 1]
        oof_preds[test_idx] = preds
        metrics.append(evaluate(y_val, preds))
        models.append(model)
        
    avg_metrics = {k: np.mean([m[k] for m in metrics]) for k in metrics[0]}
    avg_metrics["mode"] = mode
    
    # Save the best model from fold 0 as representative (for feature importance / inference)
    models[0].booster_.save_model(model_path)
    
    return avg_metrics, models[0], oof_preds

# 4. Combined Model
print("\n--- COMBINED T/E/H MODEL ---")
# Since groups are very sparse across languages, doing GroupKFold across all might be weird (4 groups total),
# but let's try it.
combined_metrics, combined_model, oof_preds = train_eval_model(
    X_teh, y_teh, groups_teh, "models/prosody/v3/lightgbm_teh_combined_v1.txt", use_group_kfold=True
)

print(json.dumps(combined_metrics, indent=2))

# Per language metrics for combined model
per_lang_results = {}
for lang in ["Tamil", "English", "Hindi"]:
    idx = np.where(langs_teh == lang)[0]
    res = evaluate(y_teh[idx], oof_preds[idx])
    res["sample_count"] = len(idx)
    res["real_count"] = int(sum(y_teh[idx]==0))
    res["synthetic_count"] = int(sum(y_teh[idx]==1))
    per_lang_results[lang] = res

with open("reports/prosody/v3/teh_combined_results.json", "w") as f:
    json.dump({"overall": combined_metrics, "per_language": per_lang_results}, f, indent=2)

# 5. Language Specific Models
print("\n--- LANGUAGE-SPECIFIC MODELS ---")
for lang in ["Tamil", "English", "Hindi"]:
    print(f"\n{lang} Model:")
    idx = np.where(langs_teh == lang)[0]
    X_lang, y_lang, groups_lang = X_teh[idx], y_teh[idx], groups_teh[idx]
    
    use_gk = len(np.unique(groups_lang)) >= 2
    metrics, _, _ = train_eval_model(
        X_lang, y_lang, groups_lang, f"models/prosody/v3/lightgbm_{lang.lower()}_v1.txt", use_group_kfold=use_gk
    )
    metrics["sample_count"] = len(idx)
    metrics["real_count"] = int(sum(y_lang==0))
    metrics["synthetic_count"] = int(sum(y_lang==1))
    metrics["group_count"] = len(np.unique(groups_lang))
    
    print(json.dumps(metrics, indent=2))
    with open(f"reports/prosody/v3/{lang.lower()}_results.json", "w") as f:
        json.dump(metrics, f, indent=2)

# 6. Feature Ablation
print("\n--- FEATURE ABLATION (Combined Dataset) ---")
with open("feature_names_v3.json", "r") as fn:
    v3_info = json.load(fn)

def get_indices(families):
    return [i for i, info in enumerate(v3_info) if info["family"] in families]

# Model 1: Acoustic
idx_1 = get_indices(["ACOUSTIC"])
m1_res, _, _ = train_eval_model(X_teh[:, idx_1], y_teh, groups_teh, "models/prosody/v3/lightgbm_ablation_m1.txt", use_group_kfold=True)

# Model 2: Acoustic + Prosodic
idx_2 = get_indices(["ACOUSTIC", "PROSODIC"])
m2_res, _, _ = train_eval_model(X_teh[:, idx_2], y_teh, groups_teh, "models/prosody/v3/lightgbm_ablation_m2.txt", use_group_kfold=True)

# Model 3: Acoustic + Prosodic + VQ
idx_3 = get_indices(["ACOUSTIC", "PROSODIC", "VOICE QUALITY"])
m3_res, _, _ = train_eval_model(X_teh[:, idx_3], y_teh, groups_teh, "models/prosody/v3/lightgbm_ablation_m3.txt", use_group_kfold=True)

# Model 4: Full (Includes Behavioural)
m4_res = combined_metrics # Same as combined model

ablation_df = pd.DataFrame([
    {"Model": "M1 (Acoustic)", **m1_res},
    {"Model": "M2 (+Prosodic)", **m2_res},
    {"Model": "M3 (+VQ)", **m3_res},
    {"Model": "M4 (+Behavioural)", **m4_res},
])
ablation_df.to_csv("reports/prosody/v3/ablation_results.csv", index=False)
print(ablation_df[["Model", "roc_auc", "accuracy", "f1"]])

# 9. Feature Importance
print("\n--- FEATURE IMPORTANCE ---")
booster = combined_model.booster_
split_imp = booster.feature_importance(importance_type='split')
gain_imp = booster.feature_importance(importance_type='gain')

v3_names = [info["name"] for info in v3_info]
v3_fams = [info["family"] for info in v3_info]

imp_df = pd.DataFrame({
    "feature": v3_names,
    "family": v3_fams,
    "split": split_imp,
    "gain": gain_imp
}).sort_values(by="gain", ascending=False)

imp_df.to_csv("reports/prosody/v3/feature_importance.csv", index=False)
print("Top 10 features by gain:")
print(imp_df.head(10))

print("\nFamily importance by gain:")
print(imp_df.groupby("family")["gain"].sum().sort_values(ascending=False))

# 12. External Audio (WhatsApp)
print("\n--- EXTERNAL WHATSAPP TEST ---")
try:
    arr, sr = sf.read("real_whatsapp.ogg")
    wa_features = extract_features_v3(arr, sr)
    wa_vec = [wa_features.get(n, 0.0) for n in v3_names]
    wa_score = combined_model.predict_proba(np.array([wa_vec]))[0, 1]
    
    wa_res = {
        "audio": "real_whatsapp.ogg",
        "synthetic_speech_score": float(wa_score),
        "predicted_class": int(wa_score >= 0.5)
    }
    print(json.dumps(wa_res, indent=2))
    with open("reports/prosody/v3/external_validation.json", "w") as f:
        json.dump(wa_res, f, indent=2)
except Exception as e:
    print(f"WhatsApp test failed: {e}")

