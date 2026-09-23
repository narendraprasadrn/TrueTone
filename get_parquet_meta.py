import urllib.request
import pandas as pd
import pyarrow.parquet as pq
import io

base_url = "https://huggingface.co/datasets/SherryT997/IndicTTS-Deepfake-Challenge-Data/resolve/main/data/train-{idx:05d}-of-00035.parquet"
dfs = []

# To make it fast, we will download a few parquet files first to verify schema and splits
for i in range(35):
    url = base_url.format(idx=i)
    # read specific columns via duckdb or pandas?
    # Actually, pandas read_parquet supports remote URLs and we can specify columns!
    print(f"Reading {url} ...")
    try:
        df = pd.read_parquet(url, columns=["id", "language", "is_tts"])
        dfs.append(df)
    except Exception as e:
        print(f"Failed {url}: {e}")

final_df = pd.concat(dfs, ignore_index=True)
final_df.to_csv("metadata_train.csv", index=False)
print("Saved metadata_train.csv")
print(f"Total rows: {len(final_df)}")
print(final_df.head())
