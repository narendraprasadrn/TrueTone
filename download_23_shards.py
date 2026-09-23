import time
import json
import pandas as pd
from huggingface_hub import hf_hub_download

shards = [0, 1, 3, 4, 6, 7, 9, 10, 12, 13, 15, 16, 18, 19, 21, 22, 24, 25, 27, 28, 30, 31, 33]
paths = []

start = time.time()
print(f"Downloading {len(shards)} shards...")

for i in shards:
    fname = f"data/train-{i:05d}-of-00035.parquet"
    print(f"Fetching {fname}...")
    try:
        p = hf_hub_download(repo_id='SherryT997/IndicTTS-Deepfake-Challenge-Data', repo_type='dataset', filename=fname)
        paths.append(p)
    except Exception as e:
        print(f"Error {i}: {e}")

print(f"Downloaded {len(paths)} shards in {time.time()-start:.1f}s")

# Now read them into a dataframe to count everything requested
dfs = []
for p in paths:
    # Just read the metadata columns
    dfs.append(pd.read_parquet(p, columns=['id', 'language', 'is_tts']))

df = pd.concat(dfs, ignore_index=True)

# Generate prefixes using the first two underscore parts, or the whole thing if less than 2
def get_prefix(x):
    parts = x.split('_')
    if len(parts) >= 3:
        return '_'.join(parts[:2]) # E.g. ASM_F or train_gujaratimale
    return x

df['prefix'] = df['id'].apply(get_prefix)

report = {
    "actual_downloaded_size_gb": sum([pd.io.common.os.path.getsize(p) for p in paths]) / (1024**3),
    "actual_row_count": len(df),
    "real_count": int((df['is_tts'] == 0).sum()),
    "synthetic_count": int((df['is_tts'] == 1).sum()),
    "languages": df['language'].unique().tolist(),
    "duplicate_count": int(df['id'].duplicated().sum()),
    "number_of_unique_grouping_prefixes": df['prefix'].nunique()
}

with open("subset_report.json", "w") as f:
    json.dump(report, f, indent=2)

print("Report generated.")
