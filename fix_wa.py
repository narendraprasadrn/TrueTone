import json
import numpy as np
import librosa
from feature_extractor import extract_features_v2
import lightgbm as lgb

with open("data/features/prosody/feature_names.json") as f:
    feature_names = json.load(f)
feature_names = [f for f in feature_names if f not in ["id", "is_tts"]]

wa_audio, wa_sr = librosa.load("truetone/backend/real_whatsapp_uncompressed.wav", sr=16000)
wa_features = extract_features_v2(wa_audio, 16000)
wa_X_old = np.array([wa_features[f] for f in feature_names[:10]]).reshape(1, -1)

current_booster = lgb.Booster(model_file="truetone/backend/app/detection/models/prosody_lgbm.txt")
pred_old = float(current_booster.predict(wa_X_old)[0])

with open("reports/prosody/whatsapp_external_check.json", "r") as f:
    data = json.load(f)
data["old_model_p_synthetic"] = pred_old
with open("reports/prosody/whatsapp_external_check.json", "w") as f:
    json.dump(data, f, indent=2)
