import pandas as pd
import glob
import os

cache_dir = os.path.expanduser("~/.cache/huggingface/hub/datasets--SherryT997--IndicTTS-Deepfake-Challenge-Data/snapshots/*/data/train-*.parquet")
all_files = glob.glob(cache_dir)
expected_shards = [0, 1, 3, 4, 6, 7, 9, 10, 12, 13, 15, 16, 18, 19, 21, 22, 24, 25, 27, 28, 30, 31, 33]
files_to_load = []
for f in all_files:
    # check if the number is in expected
    base = os.path.basename(f)
    try:
        num = int(base.split('-')[1])
        if num in expected_shards:
            files_to_load.append(f)
    except:
        pass

dfs = []
for f in files_to_load:
    dfs.append(pd.read_parquet(f, columns=['id', 'language', 'is_tts']))

df = pd.concat(dfs, ignore_index=True)
def get_prefix(x):
    parts = x.split('_')
    if len(parts) >= 3:
        return '_'.join(parts[:2])
    return x

df['group_key'] = df['id'].apply(get_prefix)

group_stats = df.groupby('group_key').agg(
    total=('id', 'count'),
    real=('is_tts', lambda x: (x==0).sum()),
    synth=('is_tts', lambda x: (x==1).sum()),
    langs=('language', lambda x: list(set(x)))
).reset_index()

print(group_stats.to_string())
print(f"\nTotal groups: {len(group_stats)}")
