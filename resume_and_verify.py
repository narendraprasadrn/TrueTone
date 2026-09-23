import os
import glob
import json
import pandas as pd
from huggingface_hub import hf_hub_download

expected_shards = [0, 1, 3, 4, 6, 7, 9, 10, 12, 13, 15, 16, 18, 19, 21, 22, 24, 25, 27, 28, 30, 31, 33]
existing_10 = [0, 1, 3, 4, 6, 7, 9, 10, 12, 13]
missing_13 = [15, 16, 18, 19, 21, 22, 24, 25, 27, 28, 30, 31, 33]

old_repo = 'SherryT997/IndicTTS-Deepfake-Challenge-Data'
new_repo = 'SherryT997/IndicTTS-Deepfake-Challenge-Data'

paths = {}
missing = []
present = []

print("1. Confirming the 10 existing shards...")
for i in existing_10:
    fname = f"data/train-{i:05d}-of-00035.parquet"
    try:
        p = hf_hub_download(repo_id=old_repo, repo_type='dataset', filename=fname, local_files_only=True)
        paths[i] = p
        present.append(i)
    except Exception as e:
        print(f"Error finding shard {i}: {e}")
        missing.append(i)

if missing:
    print(f"Error: Missing some of the 10 existing shards! {missing}")
else:
    print("All 10 existing shards found.")

print("\n2. Downloading the 13 missing shards...")
for i in missing_13:
    fname = f"data/train-{i:05d}-of-00035.parquet"
    print(f"Downloading {fname} from {new_repo}...")
    try:
        p = hf_hub_download(repo_id=new_repo, repo_type='dataset', filename=fname, resume_download=True)
        paths[i] = p
        present.append(i)
    except Exception as e:
        print(f"Error downloading shard {i}: {e}")
        missing.append(i)

print("\n3. Verifying all 23 shards...")
total_size = 0
dfs = []
corrupt = []

for i in expected_shards:
    if i in paths:
        p = paths[i]
        total_size += os.path.getsize(p)
        try:
            df = pd.read_parquet(p, columns=['id', 'language', 'is_tts'])
            dfs.append(df)
        except Exception as e:
            print(f"Error reading shard {i}: {e}")
            corrupt.append(i)
    else:
        print(f"Shard {i} is missing from paths dict.")
        if i not in missing:
            missing.append(i)

if not dfs:
    print("No dataframes loaded.")
    exit(1)

full_df = pd.concat(dfs, ignore_index=True)
actual_row_count = len(full_df)
real_count = int((full_df['is_tts'] == 0).sum())
synthetic_count = int((full_df['is_tts'] == 1).sum())
languages = full_df['language'].unique().tolist()
duplicate_count = int(full_df['id'].duplicated().sum())

print("\n--- FINAL VERIFICATION REPORT ---")
print(f"Expected shards: {len(expected_shards)}")
print(f"Present shards: {len(present)}")
print(f"Missing shards: {len(missing)}")
if missing:
    print(f"Missing list: {missing}")
print(f"Total local size: {total_size / (1024**3):.2f} GB")
print(f"Total row count: {actual_row_count}")
print(f"Real count (is_tts=0): {real_count}")
print(f"Synthetic count (is_tts=1): {synthetic_count}")
print(f"Languages represented: {len(languages)} - {languages}")
print(f"Corrupt/unreadable files: {len(corrupt)} - {corrupt}")
print(f"Duplicate IDs: {duplicate_count}")

