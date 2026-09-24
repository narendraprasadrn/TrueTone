import pandas as pd
train_df = pd.read_csv("data/splits_multilingual/internal_train.csv")
val_df = pd.read_csv("data/splits_multilingual/internal_val.csv")
pool = pd.concat([train_df, val_df])

def get_group_3(idx): return '_'.join(idx.split('_')[:3])
def get_group_4(idx): return '_'.join(idx.split('_')[:4])

for lang in ["Tamil", "English", "Hindi"]:
    subset = pool[pool['language'] == lang]
    print(f"{lang} [:3] groups:", subset['id'].apply(get_group_3).unique())
