import os
import json
import numpy as np
import librosa
import pandas as pd
import sys

# Load feature names
with open("data/features/prosody/feature_names.json") as f:
    feature_names = json.load(f)
feature_names = [x for x in feature_names if x not in ["id", "is_tts"]]

sys.path.append("truetone/backend")
from app.preprocessing.pipeline import preprocess_for_detection, prepare_aasist_context
from app.detection.aasist import AasistDetector
from app.detection.prosody import ProsodyDetector
from feature_extractor import extract_features_v2
import lightgbm as lgb

base_dir = "truetone/backend"
aasist = AasistDetector(os.path.join(base_dir, "third_party/aasist/models/weights/AASIST-L.pth"))
old_prosody = ProsodyDetector(os.path.join(base_dir, "app/detection/models/prosody_lgbm.txt"))
new_prosody_booster = lgb.Booster(model_file="models/prosody/lightgbm_indictts_v1.txt")

files_to_test = {
    "HUMAN": [
        "truetone/backend/natural.wav",
        "truetone/backend/v_audio.ogg"
    ],
    "SYNTHETIC": [
        "truetone/backend/synthetic.wav",
        "truetone/backend/synthetic_cloned.ogg"
    ],
    "WHATSAPP": [
        "truetone/backend/real_whatsapp.ogg"
    ]
}

results = []

for category, paths in files_to_test.items():
    for path in paths:
        if not os.path.exists(path):
            print(f"Skipping {path}, not found.")
            continue
            
        print(f"Processing {path} ({category})")
        audio_data, sr = librosa.load(path, sr=None)
        
        # Test bench logic
        step_duration_sec = 1.0
        aasist_window_sec = 64600 / 16000 # 4.0375
        other_window_sec = 2.0
        
        total_samples = len(audio_data)
        current_end_sec = 2.0
        window_index = 1
        
        while current_end_sec * sr <= total_samples:
            end_idx = int(current_end_sec * sr)
            
            other_start_sec = max(0.0, current_end_sec - other_window_sec)
            other_start_idx = int(other_start_sec * sr)
            other_chunk = audio_data[other_start_idx:end_idx]
            
            aasist_start_idx = other_start_idx
            aasist_end_idx = min(total_samples, aasist_start_idx + int(aasist_window_sec * sr))
            aasist_chunk = audio_data[aasist_start_idx:aasist_end_idx]
            
            # Pad aasist_chunk if needed
            if len(aasist_chunk) < int(aasist_window_sec * sr):
                aasist_chunk = np.pad(aasist_chunk, (0, int(aasist_window_sec * sr) - len(aasist_chunk)))
                
            aasist_processed_raw = preprocess_for_detection(aasist_chunk, sr, debug=False)
            aasist_processed = prepare_aasist_context(aasist_processed_raw)
            a_score = aasist.predict_aasist(aasist_processed, 16000)["spoof_score"]
            
            other_processed = preprocess_for_detection(other_chunk, sr, debug=False)
            p_old_score = old_prosody.score(other_processed, 16000)["spoof_score"]
            
            # Extract V2 features
            # Resample to 16k if needed for feature_extractor
            if sr != 16000:
                other_chunk_16k = librosa.resample(other_chunk, orig_sr=sr, target_sr=16000)
            else:
                other_chunk_16k = other_chunk
                
            try:
                feat_dict = extract_features_v2(other_chunk_16k, 16000)
                for k, v in feat_dict.items():
                    if np.isnan(v) or np.isinf(v):
                        feat_dict[k] = 0.0
                X_new = np.array([feat_dict[f] for f in feature_names]).reshape(1, -1)
                p_new_score = float(new_prosody_booster.predict(X_new)[0])
            except Exception as e:
                p_new_score = np.nan
                print(f"Error extracting features for {path} at window {window_index}: {e}")
                
            results.append({
                "category": category,
                "audio": os.path.basename(path),
                "window": f"#{window_index}",
                "time": f"{other_start_sec:.1f}-{current_end_sec:.1f}s",
                "start_time": other_start_sec,
                "end_time": current_end_sec,
                "aasist_score": a_score,
                "old_prosody_score": p_old_score,
                "new_v2_prosody_score": p_new_score
            })
            
            current_end_sec += step_duration_sec
            window_index += 1

df = pd.DataFrame(results)
os.makedirs("reports/prosody", exist_ok=True)
df.to_csv("reports/prosody/controlled_model_comparison.csv", index=False)
df.to_json("reports/prosody/controlled_model_comparison.json", orient="records", indent=2)

print("DONE")
