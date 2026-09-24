import json
import pandas as pd
import os

train_df = pd.read_csv("data/splits_multilingual/internal_train.csv")
val_df = pd.read_csv("data/splits_multilingual/internal_val.csv")
test_df = pd.read_csv("data/splits_multilingual/test.csv")
all_manifest = pd.concat([train_df, val_df, test_df]).set_index('id')

cache_file = "data/features/prosody/v3/feature_cache.jsonl"
success = 0
failures = 0
nans = 0
infs = 0
dims = set()
ids_processed = set()
duplicates = 0

real_count = 0
synth_count = 0
tamil_count = 0
eng_count = 0
hindi_count = 0

if os.path.exists(cache_file):
    with open(cache_file, "r") as f:
        for line in f:
            data = json.loads(line)
            aid = data['id']
            if aid in ids_processed:
                duplicates += 1
            ids_processed.add(aid)
            
            if data.get('error') is not None:
                failures += 1
            else:
                success += 1
                dims.add(len(data['features']))
                nans += data.get('nans', 0)
                infs += data.get('infs', 0)
                
                if aid in all_manifest.index:
                    row = all_manifest.loc[aid]
                    if isinstance(row, pd.DataFrame):
                        row = row.iloc[0]
                    if row['is_tts'] == 0:
                        real_count += 1
                    else:
                        synth_count += 1
                        
                    if row['language'] == 'Tamil':
                        tamil_count += 1
                    elif row['language'] == 'English':
                        eng_count += 1
                    elif row['language'] == 'Hindi':
                        hindi_count += 1

print("--- V3 CACHE VALIDATION ---")
print(f"Total attempted: {success + failures}")
print(f"Successful: {success}")
print(f"Failed: {failures}")
print(f"Feature dimensions: {dims}")
print(f"NaN count: {nans}")
print(f"Inf count: {infs}")
print(f"Duplicate IDs: {duplicates}")
print(f"Real count (success): {real_count}")
print(f"Synthetic count (success): {synth_count}")
print(f"Tamil count (success): {tamil_count}")
print(f"English count (success): {eng_count}")
print(f"Hindi count (success): {hindi_count}")
