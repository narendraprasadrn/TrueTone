import pandas as pd
import numpy as np
import os
import glob
from datasets import load_dataset, Audio
import json
import time
import multiprocessing
from feature_extractor import extract_features_v2

# 1. Load Manifests
train_df = pd.read_csv("data/splits_multilingual/internal_train.csv")
val_df = pd.read_csv("data/splits_multilingual/internal_val.csv")
test_df = pd.read_csv("data/splits_multilingual/test.csv")

all_manifest = pd.concat([train_df, val_df, test_df])
all_ids = set(all_manifest['id'])

cache_dir = "data/features/prosody"
os.makedirs(cache_dir, exist_ok=True)

# 2. Check Existing Cache (Resume Capability)
cache_file = os.path.join(cache_dir, "feature_cache.jsonl")
processed_ids = set()
if os.path.exists(cache_file):
    with open(cache_file, "r") as f:
        for line in f:
            data = json.loads(line)
            processed_ids.add(data['id'])
print(f"Resuming from {len(processed_ids)} already cached features.")

to_process_ids = all_ids - processed_ids

# 3. Load HuggingFace Dataset
hf_cache_dir = os.path.expanduser("~/.cache/huggingface/hub/datasets--SherryT997--IndicTTS-Deepfake-Challenge-Data/snapshots/*/data/train-*.parquet")
all_files = glob.glob(hf_cache_dir)
expected_shards = [0, 1, 3, 4, 6, 7, 9, 10, 12, 13, 15, 16, 18, 19, 21, 22, 24, 25, 27, 28, 30, 31, 33]
files_to_load = [f for f in all_files if int(os.path.basename(f).split('-')[1]) in expected_shards]

ds = load_dataset("parquet", data_files=files_to_load, split="train")
ds = ds.cast_column("audio", Audio(sampling_rate=16000))

# 4. Multiprocessing Worker
def extract_worker(item_data):
    idx, arr, sr = item_data
    try:
        if arr.ndim > 1: arr = arr.mean(axis=0)
        arr = arr.flatten()
        f_dict = extract_features_v2(arr, sr=16000)
        for k, v in f_dict.items():
            if np.isnan(v) or np.isinf(v):
                f_dict[k] = 0.0
        return (idx, f_dict, None)
    except Exception as e:
        return (idx, None, str(e))

# 5. Extraction Loop with Bounded Memory
def dataset_generator():
    # Only yield items that we need to process
    for item in ds:
        idx = item['id']
        if idx in to_process_ids:
            yield (idx, item['audio']['array'], item['audio']['sampling_rate'])

if __name__ == '__main__':
    pool = multiprocessing.Pool(processes=4)
    
    success = 0
    failed = 0
    skipped = len(processed_ids)
    
    t0 = time.time()
    
    with open(cache_file, "a") as f:
        # imap_unordered handles bounded memory well as long as we consume it
        for idx, f_dict, err in pool.imap_unordered(extract_worker, dataset_generator(), chunksize=5):
            if f_dict is not None:
                f_dict['id'] = idx
                f_dict = {k: float(v) if isinstance(v, (np.float32, np.float64, np.integer)) else v for k, v in f_dict.items()}
                f.write(json.dumps(f_dict) + "\n")
                success += 1
            else:
                failed += 1
                print(f"Failed {idx}: {err}")
                
            if (success + failed) % 1000 == 0:
                print(f"Processed {success + failed} / {len(to_process_ids)}...")
                
    pool.close()
    pool.join()
    t1 = time.time()
    
    print(f"\n--- EXTRACTION COMPLETE ---")
    print(f"Time: {t1-t0:.2f} seconds")
    print(f"Success: {success}")
    print(f"Failed: {failed}")
    print(f"Skipped (Cached): {skipped}")
    
    # 6. Build Parquet Splits
    print("\nBuilding Parquet Splits...")
    # Read back all features
    features_db = {}
    with open(cache_file, "r") as f:
        for line in f:
            data = json.loads(line)
            idx = data.pop('id')
            features_db[idx] = data
            
    if len(features_db) > 0:
        feature_names = list(next(iter(features_db.values())).keys())
        with open(os.path.join(cache_dir, "feature_names.json"), "w") as f:
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
        pd.DataFrame(records).to_parquet(os.path.join(cache_dir, f"{name}_features.parquet"))

    make_split_cache(train_df, "train")
    make_split_cache(val_df, "val")
    make_split_cache(test_df, "test")
    
    print("Splits built successfully.")
