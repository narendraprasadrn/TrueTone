import pandas as pd
import glob
import os
import json
import hashlib

def get_prefix(x):
    parts = x.split('_')
    if len(parts) >= 3:
        return '_'.join(parts[:2])
    return x

# 1. Load Data
cache_dir = os.path.expanduser("~/.cache/huggingface/hub/datasets--SherryT997--IndicTTS-Deepfake-Challenge-Data/snapshots/*/data/train-*.parquet")
all_files = glob.glob(cache_dir)
expected_shards = [0, 1, 3, 4, 6, 7, 9, 10, 12, 13, 15, 16, 18, 19, 21, 22, 24, 25, 27, 28, 30, 31, 33]
files_to_load = []
for f in all_files:
    try:
        if int(os.path.basename(f).split('-')[1]) in expected_shards:
            files_to_load.append(f)
    except: pass

dfs = []
for f in files_to_load:
    dfs.append(pd.read_parquet(f, columns=['id', 'language', 'is_tts']))
full_df = pd.concat(dfs, ignore_index=True)

full_df['group_key'] = full_df['id'].apply(get_prefix)

group_stats = full_df.groupby('group_key').agg(
    total=('id', 'count'),
    real=('is_tts', lambda x: (x==0).sum()),
    synth=('is_tts', lambda x: (x==1).sum()),
    language=('language', lambda x: x.iloc[0])
).reset_index().sort_values('total', ascending=False)

split_assignment = {}
langs_missing_test = []

for lang in group_stats['language'].unique():
    lang_groups = group_stats[group_stats['language'] == lang].to_dict('records')
    
    if len(lang_groups) == 1:
        split_assignment[lang_groups[0]['group_key']] = 'train'
        langs_missing_test.append((lang, "Only 1 group available"))
    elif len(lang_groups) == 2:
        split_assignment[lang_groups[0]['group_key']] = 'train'
        split_assignment[lang_groups[1]['group_key']] = 'test'
    else:
        # > 2 groups. Ensure at least one in Train, Val, Test.
        split_assignment[lang_groups[0]['group_key']] = 'train'
        split_assignment[lang_groups[1]['group_key']] = 'test'
        split_assignment[lang_groups[2]['group_key']] = 'val'
        
        # Remaining greedy by 80/10/10 target relative to total lang size
        total_lang_size = sum(g['total'] for g in lang_groups)
        targets = {'train': 0.8 * total_lang_size, 'val': 0.1 * total_lang_size, 'test': 0.1 * total_lang_size}
        curr = {'train': lang_groups[0]['total'], 'test': lang_groups[1]['total'], 'val': lang_groups[2]['total']}
        
        for g in lang_groups[3:]:
            # Find largest deficit
            deficits = {sp: targets[sp] - curr[sp] for sp in ['train', 'val', 'test']}
            best_sp = max(deficits, key=deficits.get)
            split_assignment[g['group_key']] = best_sp
            curr[best_sp] += g['total']

full_df['split'] = full_df['group_key'].map(split_assignment)

os.makedirs("data/splits_multilingual", exist_ok=True)
train_df = full_df[full_df['split'] == 'train']
val_df = full_df[full_df['split'] == 'val']
test_df = full_df[full_df['split'] == 'test']

train_df.drop(columns=['split']).to_csv("data/splits_multilingual/train.csv", index=False)
val_df.drop(columns=['split']).to_csv("data/splits_multilingual/val.csv", index=False)
test_df.drop(columns=['split']).to_csv("data/splits_multilingual/test.csv", index=False)

def get_stats(df):
    return {
        'total': len(df),
        'real': int((df['is_tts'] == 0).sum()),
        'synth': int((df['is_tts'] == 1).sum()),
        'languages': df['language'].unique().tolist(),
        'unique_groups': df['group_key'].nunique(),
        'lang_counts': df.groupby('language').size().to_dict()
    }

summary = {
    'train': get_stats(train_df),
    'val': get_stats(val_df),
    'test': get_stats(test_df),
    'total_rows': len(full_df),
    'langs_missing_test': langs_missing_test
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

with open("data/splits_multilingual/split_summary.json", "w") as f:
    json.dump(summary, f, indent=2)

