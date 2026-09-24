import pandas as pd
import numpy as np
import json
import lightgbm as lgb
from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score, f1_score

with open("reports/prosody/evaluation_summary.json") as f:
    eval_summary = json.load(f)

test_df = pd.read_parquet("data/features/prosody/test_features.parquet")
y_test = test_df["is_tts"].values

with open("data/features/prosody/feature_names.json") as f:
    feature_names = json.load(f)
feature_names = [f for f in feature_names if f not in ["id", "is_tts"]]

current_booster = lgb.Booster(model_file="truetone/backend/app/detection/models/prosody_lgbm.txt")
X_test_old = test_df[feature_names[:10]].values

preds_prob = current_booster.predict(X_test_old)
preds = (preds_prob > 0.5).astype(int)

eval_summary["test_current_model"] = {
    "roc_auc": roc_auc_score(y_test, preds_prob),
    "accuracy": accuracy_score(y_test, preds),
    "precision": precision_score(y_test, preds),
    "recall": recall_score(y_test, preds),
    "f1": f1_score(y_test, preds)
}

if "test_current_model_error" in eval_summary:
    del eval_summary["test_current_model_error"]

with open("reports/prosody/evaluation_summary.json", "w") as f:
    json.dump(eval_summary, f, indent=2)

# Also update per language results for old model
lang_results = []
test_manifest = pd.read_csv("data/splits_multilingual/test.csv")
merged = pd.merge(test_df, test_manifest[["id", "language"]], on="id", how="left")

for lang, group in merged.groupby("language"):
    y_lang = group["is_tts"].values
    preds_prob_old = current_booster.predict(group[feature_names[:10]].values)
    preds_old = (preds_prob_old > 0.5).astype(int)
    
    # Let's get new model preds too to regenerate the CSV correctly
    model = lgb.Booster(model_file="models/prosody/lightgbm_indictts_v1.txt")
    preds_prob_new = model.predict(group[feature_names].values)
    preds_new = (preds_prob_new > 0.5).astype(int)
    
    lang_results.append({
        "language": lang,
        "count": len(group),
        "new_auc": roc_auc_score(y_lang, preds_prob_new) if len(np.unique(y_lang)) > 1 else np.nan,
        "new_acc": accuracy_score(y_lang, preds_new),
        "new_f1": f1_score(y_lang, preds_new),
        "old_auc": roc_auc_score(y_lang, preds_prob_old) if len(np.unique(y_lang)) > 1 else np.nan,
        "old_acc": accuracy_score(y_lang, preds_old),
        "old_f1": f1_score(y_lang, preds_old),
    })

pd.DataFrame(lang_results).to_csv("reports/prosody/per_language_results.csv", index=False)
