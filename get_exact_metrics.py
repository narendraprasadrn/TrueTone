import duckdb
import pandas as pd
import json

shards = [0, 1, 3, 4, 6, 7, 9, 10, 12, 13, 15, 16, 18, 19, 21, 22, 24, 25, 27, 28, 30, 31, 33]
urls = [f'https://huggingface.co/datasets/SherryT997/IndicTTS-Deepfake-Challenge-Data/resolve/main/data/train-{i:05d}-of-00035.parquet' for i in shards]

dfs = []
print("Fetching metadata for 23 shards...")
for i, url in zip(shards, urls):
    try:
        df = duckdb.query(f"SELECT id, language, is_tts FROM read_parquet('{url}')").df()
        df['shard'] = i
        dfs.append(df)
        print(f"Shard {i} done, rows: {len(df)}")
    except Exception as e:
        print(f"Error {i}: {e}")

final_df = pd.concat(dfs, ignore_index=True)

def get_prefix(x):
    parts = x.split('_')
    if len(parts) >= 3:
        return '_'.join(parts[:2])
    return x

final_df['prefix'] = final_df['id'].apply(get_prefix)

prefix_shards = final_df.groupby('prefix')['shard'].nunique()
overlap_count = (prefix_shards > 1).sum()

report = {
    "actual_row_count": len(final_df),
    "real_count": int((final_df['is_tts'] == 0).sum()),
    "synthetic_count": int((final_df['is_tts'] == 1).sum()),
    "languages": final_df['language'].unique().tolist(),
    "duplicate_count": int(final_df['id'].duplicated().sum()),
    "unique_prefixes": int(final_df['prefix'].nunique()),
    "prefix_overlap_concerns": int(overlap_count)
}

print(json.dumps(report, indent=2))

# Also calculate approximate size from previous data
# Approx 380 MB per shard -> 8.7 GB
