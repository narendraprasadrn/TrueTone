import pandas as pd
import numpy as np
import os
import glob
from datasets import load_dataset, Audio
import json
import librosa
from feature_extractor import extract_features_v2, extract_old_10_features
import time

os.makedirs("data/features/prosody", exist_ok=True)

train_df = pd.read_csv("data/splits_multilingual/internal_train.csv")
val_df = pd.read_csv("data/splits_multilingual/internal_val.csv")
test_df = pd.read_csv("data/splits_multilingual/test.csv")

all_ids = set(train_df['id']).union(set(val_df['id'])).union(set(test_df['id']))

cache_dir = os.path.expanduser("~/.cache/huggingface/hub/datasets--SherryT997--IndicTTS-Deepfake-Challenge-Data/snapshots/*/data/train-*.parquet")
all_files = glob.glob(cache_dir)
expected_shards = [0, 1, 3, 4, 6, 7, 9, 10, 12, 13, 15, 16, 18, 19, 21, 22, 24, 25, 27, 28, 30, 31, 33]
files_to_load = [f for f in all_files if int(os.path.basename(f).split('-')[1]) in expected_shards]

ds = load_dataset("parquet", data_files=files_to_load, split="train")
ds = ds.cast_column("audio", Audio(sampling_rate=16000))

features_db = {}
failures = 0
failed_ids = []

t0 = time.time()
count = 0
for item in ds:
    idx = item['id']
    if idx in all_ids:
        try:
            arr = item['audio']['array']
            sr = item['audio']['sampling_rate']
            # Windowing exactly like TrueTone Prosody (usually they use whole file or pad to something? 
            # In AASIST it was exactly 64600. For prosody, it's usually the full audio or a very large window.
            # But the prompt says: "For the prosody model, use a consistent analysis duration/window strategy. Prefer the same general audio window convention used by TrueTone."
            # TrueTone `prosody.py` just calls it on the `audio_array`. We will use the exact same preprocessing:
            # "For audio sample: mono, 16kHz, consistent amplitude handling... do not use np.tile()".
            # I will pass the raw length.
            if arr.ndim > 1: arr = arr.mean(axis=0)
            arr = arr.flatten()
            
            f_dict = extract_features_v2(arr, sr=16000)
            
            # Check for NaN or inf
            has_nan = False
            for k, v in f_dict.items():
                if np.isnan(v) or np.isinf(v):
                    f_dict[k] = 0.0
                    has_nan = True
            
            features_db[idx] = f_dict
            
        except Exception as e:
            failures += 1
            failed_ids.append(idx)
            
        count += 1
        if count % 1000 == 0:
            print(f"Processed {count} items...")

t1 = time.time()
print(f"Extraction took {t1-t0:.2f} seconds. Failures: {failures}")

feature_names = list(next(iter(features_db.values())).keys())
with open("data/features/prosody/feature_names.json", "w") as f:
    json.dump(feature_names, f, indent=2)

def make_split_cache(df, name):
    records = []
    for _, row in df.iterrows():
        idx = row['id']
        if idx in features_db:
            r = features_db[idx].copy()
            r['id'] = idx
            r['is_tts'] = row['is_tts']
            records.append(r)
    pd.DataFrame(records).to_parquet(f"data/features/prosody/{name}_features.parquet")

make_split_cache(train_df, "train")
make_split_cache(val_df, "val")
make_split_cache(test_df, "test")

print("Feature cache saved.")
