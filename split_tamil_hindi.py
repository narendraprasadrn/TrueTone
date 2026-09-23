import pandas as pd
import glob
import os
import json
import hashlib
from collections import defaultdict

def get_prefix(x):
    parts = x.split('_')
    if len(parts) >= 3:
        return '_'.join(parts[:2])
    return x

# Read the multilingual split assignments to ensure we don't use TEST for training
train_multilingual = pd.read_csv("data/splits_multilingual/train.csv")
val_multilingual = pd.read_csv("data/splits_multilingual/val.csv")
test_multilingual = pd.read_csv("data/splits_multilingual/test.csv")

# We want to re-split ONLY from train_multilingual and val_multilingual (to avoid test_multilingual).
# WAIT: The prompt says: "Tamil TEST and Hindi TEST must remain completely unseen during training."
# The existing multilingual TEST set has Hindi (309 rows) and NO Tamil (Tamil was not in test).
# It says: "The existing multilingual TEST set must remain completely untouched. Tamil and Hindi samples from the existing TEST set MUST NOT enter training."
# "Create language-specific training/validation/test manifests for Tamil and Hindi."

# For Hindi: We can use the Hindi samples in `test_multilingual` as `hindi_test.csv`!
# For Tamil: Since Tamil is not in `test_multilingual` (it was 100% in train), we need to see if we can split its groups. If Tamil only has 1 group, `tamil_test.csv` will be empty.

# Let's inspect groups for Tamil and Hindi
all_df = pd.concat([train_multilingual, val_multilingual, test_multilingual])
tamil_df = all_df[all_df['language'] == 'Tamil']
hindi_df = all_df[all_df['language'] == 'Hindi']

print(f"Tamil Total: {len(tamil_df)}, Groups: {tamil_df['group_key'].nunique()}")
print(tamil_df.groupby(['group_key', 'is_tts']).size().unstack(fill_value=0))

print(f"\nHindi Total: {len(hindi_df)}, Groups: {hindi_df['group_key'].nunique()}")
print(hindi_df.groupby(['group_key', 'is_tts']).size().unstack(fill_value=0))

