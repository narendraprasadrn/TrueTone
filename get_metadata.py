from datasets import load_dataset
ds = load_dataset("SherryT997/IndicTTS-Deepfake-Challenge-Data", "default", split="train", streaming=True)
ds = ds.select_columns(["id", "language", "is_tts"])
print("Extracting metadata...")
import pandas as pd
data = []
for i, ex in enumerate(ds):
    data.append(ex)
    if i % 1000 == 0:
        print(f"Loaded {i}")
df = pd.DataFrame(data)
df.to_csv("metadata_train.csv", index=False)
print("Saved metadata_train.csv")
