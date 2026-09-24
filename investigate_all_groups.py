import pandas as pd
train_df = pd.read_csv("data/splits_multilingual/internal_train.csv")
val_df = pd.read_csv("data/splits_multilingual/internal_val.csv")
pool = pd.concat([train_df, val_df])

def get_group(idx): return '_'.join(idx.split('_')[:2])
pool['group'] = pool['id'].apply(get_group)

for lang in ["Tamil", "English", "Hindi"]:
    subset = pool[pool['language'] == lang]
    groups = subset['group'].unique()
    print(f"{lang} Train/Val groups ({len(groups)}):", groups)
