import pandas as pd
from datasets import load_dataset
ds = load_dataset("SherryT997/IndicTTS-Deepfake-Challenge-Data", split="train")
ds = ds.remove_columns(["audio"])
df = ds.to_pandas()
df.to_csv("truetone/backend/data/splits/train_full_meta.csv", index=False)
