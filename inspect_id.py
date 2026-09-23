import pandas as pd
import glob
import os

cache_dir = os.path.expanduser("~/.cache/huggingface/hub/datasets--SherryT997--IndicTTS-Deepfake-Challenge-Data/snapshots/*/data/*.parquet")
all_files = glob.glob(cache_dir)
if not all_files:
    print("No files found!")
else:
    sample_ids = []
    for f in all_files[:5]:
        d = pd.read_parquet(f, columns=['id', 'language', 'is_tts'])
        sample_ids.extend(d['id'].sample(min(10, len(d))).tolist())
    
    print("\nRandom Sample IDs from multiple shards:")
    for i in sample_ids:
        print(i)
