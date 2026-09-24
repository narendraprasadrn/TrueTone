import pandas as pd
import json

train_df = pd.read_csv("data/splits_multilingual/train.csv")

# We want 90/10 split by group_key.
# Group by group_key, get size and real/synth counts.
groups = train_df.groupby('group_key').agg(
    total=('id', 'count'),
    real=('is_tts', lambda x: (x==0).sum()),
    synth=('is_tts', lambda x: (x==1).sum())
).reset_index()

# Sort groups randomly but reproducibly
groups = groups.sample(frac=1, random_state=42)

val_keys = []
val_real = 0
val_synth = 0

target_real = train_df[train_df['is_tts'] == 0].shape[0] * 0.10
target_synth = train_df[train_df['is_tts'] == 1].shape[0] * 0.10

for _, row in groups.iterrows():
    if val_real < target_real or val_synth < target_synth:
        val_keys.append(row['group_key'])
        val_real += row['real']
        val_synth += row['synth']

internal_val = train_df[train_df['group_key'].isin(val_keys)]
internal_train = train_df[~train_df['group_key'].isin(val_keys)]

internal_train.to_csv("data/splits_multilingual/internal_train.csv", index=False)
internal_val.to_csv("data/splits_multilingual/internal_val.csv", index=False)

print(f"Internal Train: {len(internal_train)} samples")
print(f"Internal Val: {len(internal_val)} samples")
print(f"Internal Val Real/Synth: {val_real}/{val_synth}")
