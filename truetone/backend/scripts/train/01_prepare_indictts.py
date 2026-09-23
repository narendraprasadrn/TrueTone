import pandas as pd
from datasets import load_dataset
import os

print("Loading dataset...")
ds = load_dataset("SherryT997/IndicTTS-Deepfake-Challenge-Data", "default")
print("Dataset size:", ds)
train_ds = ds["train"]
test_ds = ds["test"]

print("\nColumns:", train_ds.column_names)
print("Audio feature schema:", train_ds.features["audio"])

labels = set()
langs = set()
for i, ex in enumerate(train_ds):
    labels.add(ex["is_tts"])
    langs.add(ex["language"])
    if i > 100: break

print("\nSample of is_tts values:", labels)
print("Sample of languages:", langs)

# Check if id has speaker info
print("\nSample IDs:")
for i in range(5):
    print(train_ds[i]["id"])

