import pandas as pd
import numpy as np
import json
import os
from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import lightgbm as lgb
import joblib
import traceback

os.makedirs("reports/prosody", exist_ok=True)
os.makedirs("models/prosody", exist_ok=True)

print("PHASE 2 & 3 - VERIFY FEATURE CACHE & LOAD TEST")
try:
    train_df = pd.read_parquet("data/features/prosody/train_features.parquet")
    val_df = pd.read_parquet("data/features/prosody/val_features.parquet")
    test_df = pd.read_parquet("data/features/prosody/test_features.parquet")
except Exception as e:
    print(f"Failed to load cache: {e}")
    exit(1)

with open("data/features/prosody/feature_names.json") as f:
    feature_names = json.load(f)

# The label column is "is_tts"
# Exclude "id" and "is_tts" from feature_names just in case
feature_names = [f for f in feature_names if f not in ["id", "is_tts"]]
dim = len(feature_names)

def verify_df(df, name):
    failed = []
    if df.shape[1] - 2 != dim:
        failed.append(f"Dimension mismatch in {name}: expected {dim}, got {df.shape[1] - 2}")
    
    nans = df[feature_names].isna().sum().sum()
    infs = np.isinf(df[feature_names]).sum().sum()
    
    return {
        "split": name,
        "rows": len(df),
        "real_count": int((df["is_tts"] == 0).sum()),
        "synthetic_count": int((df["is_tts"] == 1).sum()),
        "dim": dim,
        "nans": int(nans),
        "infs": int(infs),
        "failed_checks": failed
    }

verif = {
    "train": verify_df(train_df, "train"),
    "val": verify_df(val_df, "val"),
    "test": verify_df(test_df, "test"),
    "feature_names": feature_names,
    "total_rows": len(train_df) + len(val_df) + len(test_df)
}

with open("reports/prosody/cache_verification.json", "w") as f:
    json.dump(verif, f, indent=2)

print("PHASE 4 - TRAIN LIGHTGBM")
X_train = train_df[feature_names].values
y_train = train_df["is_tts"].values

X_val = val_df[feature_names].values
y_val = val_df["is_tts"].values

X_test = test_df[feature_names].values
y_test = test_df["is_tts"].values
test_ids = test_df["id"].values

# Train LightGBM
# Objective: binary (P(is_tts=1) = P(synthetic))
model = lgb.LGBMClassifier(
    objective="binary",
    metric="auc",
    learning_rate=0.03,
    num_leaves=31,
    max_depth=-1,
    min_child_samples=20,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_alpha=0.1,
    reg_lambda=0.1,
    n_estimators=1000,
    random_state=42
)

model.fit(
    X_train, y_train,
    eval_set=[(X_val, y_val)],
    callbacks=[lgb.early_stopping(50)]
)

model_path = "models/prosody/lightgbm_indictts_v1.txt"
model.booster_.save_model(model_path)

print("PHASE 5 - EVALUATE NEW MODEL")
def eval_model(model_booster, X, y, threshold=0.5):
    preds_prob = model_booster.predict(X)
    preds = (preds_prob > threshold).astype(int)
    return {
        "roc_auc": roc_auc_score(y, preds_prob),
        "accuracy": accuracy_score(y, preds),
        "precision": precision_score(y, preds),
        "recall": recall_score(y, preds),
        "f1": f1_score(y, preds),
        "confusion_matrix": confusion_matrix(y, preds).tolist(),
        "threshold": threshold
    }

eval_summary = {
    "val": eval_model(model.booster_, X_val, y_val),
    "test": eval_model(model.booster_, X_test, y_test)
}

print("PHASE 6 - COMPARE AGAINST CURRENT MODEL")
current_model_path = "truetone/backend/app/detection/models/prosody_lgbm.txt"
current_booster = lgb.Booster(model_file=current_model_path)
# Wait, current production model only uses 10 features!
# I have preserved the first 10 features in extract_features_v2
current_feature_names = current_booster.feature_name()
# Assuming the first 10 features match exactly or we can extract them
try:
    X_test_old = test_df[current_feature_names].values
    eval_summary["test_current_model"] = eval_model(current_booster, X_test_old, y_test)
except Exception as e:
    eval_summary["test_current_model_error"] = str(e)

with open("reports/prosody/evaluation_summary.json", "w") as f:
    json.dump(eval_summary, f, indent=2)

# Per language
test_manifest = pd.read_csv("data/splits_multilingual/test.csv")
merged = pd.merge(test_df, test_manifest[["id", "language"]], on="id", how="left")

lang_results = []
for lang, group in merged.groupby("language"):
    X_lang = group[feature_names].values
    y_lang = group["is_tts"].values
    
    # New model
    res_new = eval_model(model.booster_, X_lang, y_lang)
    
    # Old model
    try:
        res_old = eval_model(current_booster, group[current_feature_names].values, y_lang)
    except:
        res_old = {"roc_auc": 0, "accuracy": 0, "f1": 0}
        
    lang_results.append({
        "language": lang,
        "count": len(group),
        "new_auc": res_new["roc_auc"],
        "new_acc": res_new["accuracy"],
        "new_f1": res_new["f1"],
        "old_auc": res_old["roc_auc"],
        "old_acc": res_old["accuracy"],
        "old_f1": res_old["f1"],
    })

pd.DataFrame(lang_results).to_csv("reports/prosody/per_language_results.csv", index=False)

print("PHASE 7 - FEATURE IMPORTANCE")
importance = model.booster_.feature_importance(importance_type="gain")
imp_df = pd.DataFrame({"feature": feature_names, "gain": importance})
imp_df = imp_df.sort_values(by="gain", ascending=False)
imp_df.head(20).to_csv("reports/prosody/feature_importance.csv", index=False)

print("PHASE 8 - WHATSAPP EXTERNAL CHECK")
# Load WhatsApp sample
import librosa
from feature_extractor import extract_features_v2
wa_audio, wa_sr = librosa.load("truetone/backend/real_whatsapp_uncompressed.wav", sr=16000)
wa_features = extract_features_v2(wa_audio, 16000)
# Convert dict to array in the correct order
wa_X = np.array([wa_features[f] for f in feature_names]).reshape(1, -1)

pred_new = model.booster_.predict(wa_X)[0]
try:
    wa_X_old = np.array([wa_features[f] for f in current_feature_names]).reshape(1, -1)
    pred_old = current_booster.predict(wa_X_old)[0]
except:
    pred_old = -1.0

wa_res = {
    "new_model_p_synthetic": float(pred_new),
    "new_model_predicted_class": "SYNTHETIC" if pred_new > 0.5 else "REAL",
    "old_model_p_synthetic": float(pred_old),
    "extraction_success": True
}

with open("reports/prosody/whatsapp_external_check.json", "w") as f:
    json.dump(wa_res, f, indent=2)

print("DONE")
