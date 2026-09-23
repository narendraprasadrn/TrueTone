import time
import os
from huggingface_hub import hf_hub_download

shards = [0, 1, 3, 4, 6, 7, 9, 10, 12, 13, 15, 16, 18, 19, 21, 22, 24, 25, 27, 28, 30, 31, 33]
paths = []
start = time.time()
for i in shards:
    fname = f"data/train-{i:05d}-of-00035.parquet"
    p = hf_hub_download(repo_id='SherryT997/IndicTTS-Deepfake-Challenge-Data', repo_type='dataset', filename=fname)
    paths.append(p)

size_bytes = sum([os.path.getsize(p) for p in paths])
print(f"Downloaded {size_bytes / (1024**3):.2f} GB")
