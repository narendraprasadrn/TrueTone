import pandas as pd
import numpy as np
import json
import librosa
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, accuracy_score
import lightgbm as lgb
import os

os.makedirs("reports/prosody", exist_ok=True)

print("1. AUDIT FAILED SAMPLES")
failed_ids = ["ASM_M_INDIC_00295", "BEN_F_NAMES_01441", "BEN_M_SURPRISE_00464", "BRX_M_WIKI_01181", "NEP_F_NEWS_00633", "TAM_F_HAPPY_00096"]
manifests = {
    "train": pd.read_csv("data/splits_multilingual/internal_train.csv"),
    "val": pd.read_csv("data/splits_multilingual/internal_val.csv"),
    "test": pd.read_csv("data/splits_multilingual/test.csv")
}

failed_data = []
for fid in failed_ids:
    for split_name, df in manifests.items():
        match = df[df["id"] == fid]
        if not match.empty:
            row = match.iloc[0]
            failed_data.append({
                "id": fid,
                "split": split_name,
                "language": row["language"],
                "is_tts": int(row["is_tts"]),
                "exception": "when mode='interp', width=9 cannot exceed data.shape[axis]",
                "short_audio": True
            })

pd.DataFrame(failed_data).to_csv("reports/prosody/failed_samples.csv", index=False)

print("2. VERIFY LOCKED TEST INTEGRITY")
train_ids = set(manifests["train"]["id"])
val_ids = set(manifests["val"]["id"])
test_ids = set(manifests["test"]["id"])

overlap_train_val = len(train_ids.intersection(val_ids))
overlap_train_test = len(train_ids.intersection(test_ids))
overlap_val_test = len(val_ids.intersection(test_ids))
all_ids = list(train_ids) + list(val_ids) + list(test_ids)
duplicates = len(all_ids) - len(set(all_ids))

integrity = {
    "overlap_train_val": overlap_train_val,
    "overlap_train_test": overlap_train_test,
    "overlap_val_test": overlap_val_test,
    "duplicates": duplicates
}

print("3. VERIFY LABEL DIRECTION")
val_df = pd.read_parquet("data/features/prosody/val_features.parquet")
with open("data/features/prosody/feature_names.json") as f:
    feature_names = [x for x in json.load(f) if x not in ["id", "is_tts"]]
    
new_booster = lgb.Booster(model_file="models/prosody/lightgbm_indictts_v1.txt")

label_verify = []
# Pick 2 real and 2 synth
samples_to_check = val_df.groupby("is_tts").head(2)
for _, row in samples_to_check.iterrows():
    X = row[feature_names].values.astype(float).reshape(1, -1)
    p_synth = float(new_booster.predict(X)[0])
    label_verify.append({
        "id": row["id"],
        "true_label": int(row["is_tts"]),
        "p_synthetic": p_synth,
        "predicted_label": 1 if p_synth > 0.5 else 0
    })

print("4. INSPECT PER-LANGUAGE RESULTS")
lang_df = pd.read_csv("reports/prosody/per_language_results.csv")
# Calculate Precision/Recall from Test Set per language for the new model if not present
test_df = pd.read_parquet("data/features/prosody/test_features.parquet")
merged = pd.merge(test_df, manifests["test"][["id", "language"]], on="id", how="left")

lang_metrics = []
for lang, group in merged.groupby("language"):
    y = group["is_tts"].values
    preds_prob = new_booster.predict(group[feature_names].values)
    preds = (preds_prob > 0.5).astype(int)
    lang_metrics.append({
        "language": lang,
        "count": len(group),
        "roc_auc": roc_auc_score(y, preds_prob) if len(np.unique(y)) > 1 else None,
        "accuracy": accuracy_score(y, preds),
        "precision": precision_score(y, preds, zero_division=0),
        "recall": recall_score(y, preds, zero_division=0),
        "f1": f1_score(y, preds, zero_division=0)
    })
pd.DataFrame(lang_metrics).to_csv("reports/prosody/per_language_results.csv", index=False)

print("5. THRESHOLD CHECK")
y_val = val_df["is_tts"].values
preds_prob_val = new_booster.predict(val_df[feature_names].values)

thresholds = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
thresh_results = []
for t in thresholds:
    p = (preds_prob_val > t).astype(int)
    thresh_results.append({
        "threshold": t,
        "precision": precision_score(y_val, p, zero_division=0),
        "recall": recall_score(y_val, p, zero_division=0),
        "f1": f1_score(y_val, p, zero_division=0),
        "false_positives": int(( (p == 1) & (y_val == 0) ).sum()),
        "false_negatives": int(( (p == 0) & (y_val == 1) ).sum())
    })
pd.DataFrame(thresh_results).to_csv("reports/prosody/threshold_analysis.csv", index=False)

print("6. END-TO-END TEST WITH NEW MODEL")
import sys
sys.path.append("truetone/backend")
from app.preprocessing.pipeline import preprocess_for_detection, prepare_aasist_context
from app.detection.aasist import AasistDetector
from app.detection.prosody import ProsodyDetector
from app.detection.speaker_verification import SpeakerVerifier
from app.risk_engine.fusion import RiskEngine

base_dir = "truetone/backend"
aasist = AasistDetector(os.path.join(base_dir, "third_party/aasist/models/weights/AASIST-L.pth"))
sv = SpeakerVerifier(savedir=os.path.join(base_dir, "pretrained_models/ecapa-tdnn"))
re = RiskEngine(os.path.join(base_dir, "app/risk_engine/config.yaml"))
old_prosody = ProsodyDetector(os.path.join(base_dir, "app/detection/models/prosody_lgbm.txt"))
new_prosody = ProsodyDetector("models/prosody/lightgbm_indictts_v1.txt")

test_files = [
    ("s_audio.ogg", "truetone/frontend/public/s_audio.ogg"), # if available
    ("real_whatsapp.ogg", "truetone/backend/real_whatsapp.ogg"),
    ("real_whatsapp_uncompressed.wav", "truetone/backend/real_whatsapp_uncompressed.wav")
]

# We also need a known real and synthetic from dataset
real_id = val_df[val_df["is_tts"] == 0].iloc[0]["id"]
synth_id = val_df[val_df["is_tts"] == 1].iloc[0]["id"]

# Look them up in huggingface dataset cache
import glob
from datasets import load_dataset, Audio
hf_cache_dir = os.path.expanduser("~/.cache/huggingface/hub/datasets--SherryT997--IndicTTS-Deepfake-Challenge-Data/snapshots/*/data/train-*.parquet")
all_files = glob.glob(hf_cache_dir)
expected_shards = [0, 1, 3, 4, 6, 7, 9, 10, 12, 13, 15, 16, 18, 19, 21, 22, 24, 25, 27, 28, 30, 31, 33]
files_to_load = [f for f in all_files if int(os.path.basename(f).split('-')[1]) in expected_shards]
ds = load_dataset("parquet", data_files=files_to_load, split="train")
ds = ds.cast_column("audio", Audio(sampling_rate=16000))

found = 0
for item in ds:
    if item["id"] == real_id:
        test_files.append(("known_real", item["audio"]))
        found += 1
    if item["id"] == synth_id:
        test_files.append(("known_synth", item["audio"]))
        found += 1
    if found == 2:
        break

integration_results = []
for name, source in test_files:
    try:
        if isinstance(source, str):
            if not os.path.exists(source): continue
            audio, sr = librosa.load(source, sr=16000)
        else:
            audio = source["array"]
            sr = source["sampling_rate"]
            
        if audio.ndim > 1: audio = audio.mean(axis=0)
        audio = audio.flatten()
        
        # We need a 4s chunk for AASIST and whatever for others.
        chunk = audio[:64600] if len(audio) >= 64600 else np.pad(audio, (0, 64600 - len(audio)))
        
        aasist_input = prepare_aasist_context(preprocess_for_detection(chunk, 16000, False))
        a_score = aasist.predict_aasist(aasist_input, 16000)["spoof_score"]
        
        other_input = preprocess_for_detection(audio, 16000, False)
        p_old_score = old_prosody.score(other_input, 16000)["spoof_score"]
        
        # New Prosody detector extracts 109 features now.
        # Wait, app/detection/prosody.py extract_features only extracts 10!
        # The true "new" prosody model will fail if we use the old class!
        from feature_extractor import extract_features_v2
        feat_dict = extract_features_v2(audio, 16000)
        for k, v in feat_dict.items():
            if np.isnan(v) or np.isinf(v):
                feat_dict[k] = 0.0
        X_new = np.array([feat_dict[f] for f in feature_names]).reshape(1, -1)
        p_new_score = float(new_booster.predict(X_new)[0])
        
        ctx_flags = {"unknown_caller": False, "high_value_keywords": False, "ivr_allowlisted": False}
        win_res = re.score_window("test", a_score, p_new_score, None, ctx_flags)
        
        integration_results.append({
            "name": name,
            "aasist_score": a_score,
            "old_prosody_score": p_old_score,
            "new_prosody_score": p_new_score,
            "fused_score_with_new": win_res.r_final,
            "classification": win_res.classification
        })
    except Exception as e:
        print(f"Failed end-to-end for {name}: {e}")

with open("reports/prosody/integration_test.json", "w") as f:
    json.dump(integration_results, f, indent=2)

with open("reports/prosody/final_audit.json", "w") as f:
    json.dump({
        "integrity": integrity,
        "label_direction": label_verify
    }, f, indent=2)

print("DONE")
