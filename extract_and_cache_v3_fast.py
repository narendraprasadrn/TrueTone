import pandas as pd
import numpy as np
import os
import glob
from datasets import load_dataset, Audio
import json
from feature_extractor_v3 import extract_features_v3

def process_batch(batch):
    results_json = []
    for aid, arr, sr in zip(batch['id'], [x['array'] for x in batch['audio']], [x['sampling_rate'] for x in batch['audio']]):
        try:
            feats = extract_features_v3(arr, sr)
            nans = sum(1 for v in feats.values() if np.isnan(v))
            infs = sum(1 for v in feats.values() if np.isinf(v))
            res = {'id': aid, 'features': {k: float(v) for k, v in feats.items()}, 'nans': nans, 'infs': infs, 'error': None}
        except Exception as e:
            res = {'id': aid, 'error': str(e)}
        results_json.append(json.dumps(res))
    return {'result': results_json}

if __name__ == '__main__':
    train_df = pd.read_csv("data/splits_multilingual/internal_train.csv")
    val_df = pd.read_csv("data/splits_multilingual/internal_val.csv")
    test_df = pd.read_csv("data/splits_multilingual/test.csv")
    all_manifest = pd.concat([train_df, val_df, test_df])
    
    # Just to be safe, filter the dataframe to the expected 20,270 extracted rows + 6 failures
    all_ids = set(all_manifest['id'])
    failed_ids = {'ASM_M_INDIC_00295', 'BEN_F_NAMES_01441', 'BEN_M_SURPRISE_00464', 'BRX_M_WIKI_01181', 'NEP_F_NEWS_00633', 'TAM_F_HAPPY_00096'}
    all_ids = all_ids.union(failed_ids)

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
    
    # Extremely fast filter by ID using vectorized map or simply passing a fast function
    # Note: `ds.filter` does NOT decode audio if you don't use it in the lambda.
    ds_filtered = ds.filter(lambda batch: [i in to_process_ids for i in batch['id']], batched=True, batch_size=1000, num_proc=16)
    
    ds_filtered = ds_filtered.cast_column("audio", Audio(sampling_rate=16000))
    
    # Now run the mapping
    ds_results = ds_filtered.map(process_batch, batched=True, batch_size=100, num_proc=16, remove_columns=ds_filtered.column_names)
    
    with open(cache_file, "a") as f:
        for item in ds_results:
            f.write(item['result'] + "\n")
            
    print("Done")
