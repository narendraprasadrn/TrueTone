import pandas as pd
import glob
import os
import json
import hashlib

# 1. Load metadata from ALL 23 local Parquet shards
cache_dir = os.path.expanduser("~/.cache/huggingface/hub/datasets--SherryT997--IndicTTS-Deepfake-Challenge-Data/snapshots/*/data/train-*.parquet")
all_files = glob.glob(cache_dir)
expected_shards = [0, 1, 3, 4, 6, 7, 9, 10, 12, 13, 15, 16, 18, 19, 21, 22, 24, 25, 27, 28, 30, 31, 33]
files_to_load = []
for f in all_files:
    try:
        num = int(os.path.basename(f).split('-')[1])
        if num in expected_shards:
            files_to_load.append(f)
    except:
        pass

print(f"Loading {len(files_to_load)} files...")
dfs = []
for f in files_to_load:
    df = pd.read_parquet(f)
    # We only need metadata columns, not audio data arrays to keep it light
    cols = ['id', 'language', 'is_tts']
    if 'path' in df.columns: cols.append('path')
    # If audio is dict with path, extract it if needed, but we just keep id, language, is_tts
    dfs.append(df[cols])

full_df = pd.concat(dfs, ignore_index=True)
print(f"Total rows loaded: {len(full_df)}")

# 3. Create grouping key
def get_prefix(x):
    parts = x.split('_')
    if len(parts) >= 3:
        return '_'.join(parts[:2])
    return x

full_df['group_key'] = full_df['id'].apply(get_prefix)

# 4. Stratify into 80/10/10
group_stats = full_df.groupby('group_key').agg(
    total=('id', 'count'),
    real=('is_tts', lambda x: (x==0).sum()),
    synth=('is_tts', lambda x: (x==1).sum()),
    language=('language', lambda x: x.iloc[0])
).reset_index()

# Sort groups descending by total size to assign large blocks first
group_stats = group_stats.sort_values('total', ascending=False)

split_assignment = {} # group_key -> split

for lang in group_stats['language'].unique():
    lang_groups = group_stats[group_stats['language'] == lang]
    
    target_train_real = lang_groups['real'].sum() * 0.8
    target_train_synth = lang_groups['synth'].sum() * 0.8
    target_val_real = lang_groups['real'].sum() * 0.1
    target_val_synth = lang_groups['synth'].sum() * 0.1
    target_test_real = lang_groups['real'].sum() * 0.1
    target_test_synth = lang_groups['synth'].sum() * 0.1
    
    curr = {
        'train': {'real': 0, 'synth': 0},
        'val': {'real': 0, 'synth': 0},
        'test': {'real': 0, 'synth': 0}
    }
    targets = {
        'train': {'real': target_train_real, 'synth': target_train_synth},
        'val': {'real': target_val_real, 'synth': target_val_synth},
        'test': {'real': target_test_real, 'synth': target_test_synth}
    }
    
    for _, row in lang_groups.iterrows():
        gkey = row['group_key']
        r = row['real']
        s = row['synth']
        
        # Calculate scores for each split
        scores = {}
        for sp in ['train', 'val', 'test']:
            # How much we need
            def_r = max(0, targets[sp]['real'] - curr[sp]['real'])
            def_s = max(0, targets[sp]['synth'] - curr[sp]['synth'])
            # Normalize to avoid favoring real over synth just because of count differences
            norm_r = def_r / max(1, targets[sp]['real'])
            norm_s = def_s / max(1, targets[sp]['synth'])
            scores[sp] = norm_r + norm_s
            
        best_sp = max(scores, key=scores.get)
        split_assignment[gkey] = best_sp
        curr[best_sp]['real'] += r
        curr[best_sp]['synth'] += s

full_df['split'] = full_df['group_key'].map(split_assignment)

# 7. Save manifests
os.makedirs("data/splits", exist_ok=True)
train_df = full_df[full_df['split'] == 'train']
val_df = full_df[full_df['split'] == 'val']
test_df = full_df[full_df['split'] == 'test']

train_df.drop(columns=['split']).to_csv("data/splits/train.csv", index=False)
val_df.drop(columns=['split']).to_csv("data/splits/val.csv", index=False)
test_df.drop(columns=['split']).to_csv("data/splits/test.csv", index=False)

# Verification & Summary
def get_stats(df):
    return {
        'total': len(df),
        'real': int((df['is_tts'] == 0).sum()),
        'synth': int((df['is_tts'] == 1).sum()),
        'languages': df['language'].unique().tolist(),
        'unique_groups': df['group_key'].nunique()
    }

summary = {
    'train': get_stats(train_df),
    'val': get_stats(val_df),
    'test': get_stats(test_df),
    'total_rows': len(full_df)
}

train_groups = set(train_df['group_key'])
val_groups = set(val_df['group_key'])
test_groups = set(test_df['group_key'])

summary['group_overlap'] = {
    'train_val': len(train_groups & val_groups),
    'train_test': len(train_groups & test_groups),
    'val_test': len(val_groups & test_groups)
}

train_ids = set(train_df['id'])
val_ids = set(val_df['id'])
test_ids = set(test_df['id'])

summary['id_overlap'] = {
    'train_val': len(train_ids & val_ids),
    'train_test': len(train_ids & test_ids),
    'val_test': len(val_ids & test_ids)
}

with open("data/splits/split_summary.json", "w") as f:
    json.dump(summary, f, indent=2)

def calc_sha(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        h.update(f.read())
    return h.hexdigest()

print("\n--- SPLIT MANIFEST CHECKSUMS ---")
print(f"train.csv: {calc_sha('data/splits/train.csv')}")
print(f"val.csv: {calc_sha('data/splits/val.csv')}")
print(f"test.csv: {calc_sha('data/splits/test.csv')}")
print(f"split_summary.json: {calc_sha('data/splits/split_summary.json')}")

