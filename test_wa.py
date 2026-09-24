import librosa
import json
import numpy as np
import lightgbm as lgb
from feature_extractor_v3 import extract_features_v3

with open("feature_names_v3.json", "r") as fn:
    v3_names = [r["name"] for r in json.load(fn)]

arr, sr = librosa.load("truetone/backend/real_whatsapp_uncompressed.wav", sr=16000)
wa_features = extract_features_v3(arr, sr)
wa_vec = [wa_features.get(n, 0.0) for n in v3_names]

booster = lgb.Booster(model_file="models/prosody/v3/lightgbm_teh_combined_v1.txt")
wa_score = booster.predict(np.array([wa_vec]))[0]

wa_res = {
    "audio": "real_whatsapp_uncompressed.wav",
    "synthetic_speech_score": float(wa_score),
    "predicted_class": int(wa_score >= 0.5)
}
print(json.dumps(wa_res, indent=2))
with open("reports/prosody/v3/external_validation.json", "w") as f:
    json.dump(wa_res, f, indent=2)
