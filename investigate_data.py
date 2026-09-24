import pandas as pd
import json
import numpy as np

# Load splits
train_df = pd.read_csv("data/splits_multilingual/internal_train.csv")
val_df = pd.read_csv("data/splits_multilingual/internal_val.csv")
test_df = pd.read_csv("data/splits_multilingual/test.csv")

# Create group column (first two parts of ID)
def get_group(idx): return '_'.join(idx.split('_')[:2])
for df in [train_df, val_df, test_df]:
    df['group'] = df['id'].apply(get_group)

# Train pool vs test pool
train_pool = pd.concat([train_df, val_df])

print("--- LEAKAGE CHECK ---")
print("Train IDs:", len(train_pool))
print("Test IDs:", len(test_df))
common_ids = set(train_pool['id']).intersection(set(test_df['id']))
print("Overlapping IDs:", len(common_ids))

train_groups = set(train_pool['group'])
test_groups = set(test_df['group'])
common_groups = train_groups.intersection(test_groups)
print("Overlapping Groups:", len(common_groups))
if len(common_groups) > 0:
    print("WARNING: These groups leak from train to test:", common_groups)

print("\n--- ENGLISH GROUP DISTRIBUTION ---")
eng = pd.concat([train_pool, test_df])
eng = eng[eng['language'] == 'English']
for grp in eng['group'].unique():
    subset = eng[eng['group'] == grp]
    print(f"Group {grp}: {len(subset)} total, {sum(subset['is_tts']==0)} real, {sum(subset['is_tts']==1)} synth")
