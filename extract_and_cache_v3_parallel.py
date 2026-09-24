import pandas as pd
import numpy as np
import os
import glob
from datasets import load_dataset, Audio
import json
import time
from joblib import Parallel, delayed
from feature_extractor_v3 import extract_features_v3

def process_item(item):
    aid = item['id']
    try:
        arr = item['audio']['array']
        sr = item['audio']['sampling_rate']
        feats = extract_features_v3(arr, sr)
        nans = sum(1 for v in feats.values() if np.isnan(v))
        infs = sum(1 for v in feats.values() if np.isinf(v))
        return {'id': aid, 'features': feats, 'nans': nans, 'infs': infs, 'error': None}
    except Exception as e:
        return {'id': aid, 'error': str(e)}

if __name__ == '__main__':
    train_df = pd.read_csv("data/splits_multilingual/internal_train.csv")
    val_df = pd.read_csv("data/splits_multilingual/internal_val.csv")
    test_df = pd.read_csv("data/splits_multilingual/test.csv")
    all_manifest = pd.concat([train_df, val_df, test_df])
    all_ids = set(all_manifest['id'])

    cache_dir = "data/features/prosody/v3"
    os.makedirs(cache_dir, exist_ok=True)
    cache_file = os.path.join(cache_dir, "feature_cache.jsonl")

    processed_ids = set()
    if os.path.exists(cache_file):
        with open(cache_file, "r") as f:
            for line in f:
                processed_ids.add(json.loads(line)['id'])

    to_process_ids = all_ids - processed_ids
    if len(to_process_ids) == 0:
        print("All features extracted!")
        exit(0)

    hf_cache_dir = os.path.expanduser("~/.cache/huggingface/hub/datasets--SherryT997--IndicTTS-Deepfake-Challenge-Data/snapshots/*/data/train-*.parquet")
    all_files = glob.glob(hf_cache_dir)
    expected_shards = [0, 1, 3, 4, 6, 7, 9, 10, 12, 13, 15, 16, 18, 19, 21, 22, 24, 25, 27, 28, 30, 31, 33]
    files_to_load = [f for f in all_files if int(os.path.basename(f).split('-')[1]) in expected_shards]

    ds = load_dataset("parquet", data_files=files_to_load, split="train")
    ds = ds.cast_column("audio", Audio(sampling_rate=16000))

    # Just process everything sequentially for safety, but we can do parallel mapping 
    # if we batch them. Actually, processing sequentially on 20k rows might take 15 mins.
    # Let's filter first
    ds_filtered = ds.filter(lambda x: x['id'] in to_process_ids)
    print(f"Filtered {len(ds_filtered)} items.")
    
    with open(cache_file, "a") as f:
        # joblib parallel
        results = Parallel(n_jobs=-1, batch_size=10)(delayed(process_item)(item) for item in ds_filtered)
        for res in results:
            f.write(json.dumps(res) + "\n")
            
    print("Done")
