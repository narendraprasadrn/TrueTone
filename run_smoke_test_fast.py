import pandas as pd
import numpy as np
import os
import glob
from datasets import load_dataset, Audio
import json
import traceback
import sys

from feature_extractor import extract_features_v2
from feature_extractor_v3 import extract_features_v3

train_df = pd.read_csv("data/splits_multilingual/internal_train.csv")
test_df = pd.read_csv("data/splits_multilingual/test.csv")
full_df = pd.concat([train_df, test_df])

failed_ids = ['ASM_M_INDIC_00295', 'BEN_F_NAMES_01441', 'BEN_M_SURPRISE_00464', 'BRX_M_WIKI_01181', 'NEP_F_NEWS_00633', 'TAM_F_HAPPY_00096']
normal_ids = full_df[full_df['language'].isin(['Tamil', 'English', 'Hindi'])].head(20)['id'].tolist()
real_ids = full_df[full_df['is_tts'] == 0].head(2)['id'].tolist()
synth_ids = full_df[full_df['is_tts'] == 1].head(2)['id'].tolist()

target_ids = set(failed_ids + normal_ids + real_ids + synth_ids)
found_items = []

hf_cache_dir = os.path.expanduser("~/.cache/huggingface/hub/datasets--SherryT997--IndicTTS-Deepfake-Challenge-Data/snapshots/*/data/train-*.parquet")
all_files = glob.glob(hf_cache_dir)
expected_shards = [0, 1, 3, 4, 6, 7, 9, 10, 12, 13, 15, 16, 18, 19, 21, 22, 24, 25, 27, 28, 30, 31, 33]
files_to_load = [f for f in all_files if int(os.path.basename(f).split('-')[1]) in expected_shards]

ds = load_dataset("parquet", data_files=files_to_load, split="train")
ds = ds.cast_column("audio", Audio(sampling_rate=16000))

print(f"Scanning for {len(target_ids)} items...")
for item in ds:
    if item['id'] in target_ids:
        found_items.append(item)
        target_ids.remove(item['id'])
    if not target_ids:
        break

print(f"Found {len(found_items)} samples for testing.")

results = []
failures = []
diffs = []

for item in found_items:
    aid = item['id']
    arr = item['audio']['array']
    sr = item['audio']['sampling_rate']
    
    try:
        v3 = extract_features_v3(arr, sr)
        nans = sum(1 for v in v3.values() if np.isnan(v))
        infs = sum(1 for v in v3.values() if np.isinf(v))
        results.append({'id': aid, 'dims': len(v3), 'nans': nans, 'infs': infs})
        
        if aid in normal_ids:
            v2 = extract_features_v2(arr, sr)
            for k in v2:
                if k in v3:
                    diff = abs(v2[k] - v3[k])
                    if diff > 1e-4:
                        diffs.append({'id': aid, 'feature': k, 'v2': v2[k], 'v3': v3[k], 'diff': diff})
    except Exception as e:
        failures.append({'id': aid, 'error': str(e)})

print(f"\n--- SMOKE TEST RESULTS ---")
print(f"Total processed: {len(results)}")
print(f"Failed: {len(failures)}")
for f in failures:
    print(f"  {f['id']}: {f['error']}")
    
if len(results) > 0:
    print(f"Dimensions consistent: {all(r['dims'] == 112 for r in results)}")
    print(f"NaNs found: {sum(r['nans'] for r in results)}")
    print(f"Infs found: {sum(r['infs'] for r in results)}")

print(f"\n--- REGRESSION Diffs (V2 vs V3 for old features) ---")
print(f"Total diffs: {len(diffs)}")
if diffs:
    print("Top 5 diffs:")
    for d in diffs[:5]:
        print(f"  {d['id']} - {d['feature']}: V2={d['v2']} vs V3={d['v3']} (diff={d['diff']})")

failed_tested = [r['id'] for r in results if r['id'] in failed_ids]
print(f"\nFailed IDs successfully processed with V3: {len(failed_tested)}/6")

