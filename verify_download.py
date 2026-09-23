import os
import glob

# The HF cache path for this dataset
cache_dir = os.path.expanduser("~/.cache/huggingface/hub/datasets--SherryT997--IndicTTS-Deepfake-Challenge-Data/snapshots")

# Find the snapshot folder
snapshot_dirs = glob.glob(f"{cache_dir}/*")
if not snapshot_dirs:
    print("Snapshot not found.")
    exit(1)

snapshot_dir = snapshot_dirs[0]
data_dir = os.path.join(snapshot_dir, "data")

expected_shards = [0, 1, 3, 4, 6, 7, 9, 10, 12, 13, 15, 16, 18, 19, 21, 22, 24, 25, 27, 28, 30, 31, 33]
missing = []
present = []
total_size = 0

for i in expected_shards:
    fname = f"train-{i:05d}-of-00035.parquet"
    fpath = os.path.join(data_dir, fname)
    if os.path.exists(fpath):
        present.append(fname)
        total_size += os.path.getsize(fpath)
    else:
        missing.append(fname)

print(f"Total Expected: {len(expected_shards)}")
print(f"Present: {len(present)}")
print(f"Missing: {len(missing)}")
print(f"Size of present: {total_size / (1024**3):.2f} GB")
print(f"Missing shards: {missing}")

if len(missing) > 0:
    print("STATUS: INCOMPLETE")
else:
    print("STATUS: COMPLETE")
