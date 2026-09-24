import pandas as pd
train_df = pd.read_csv("data/splits_multilingual/internal_train.csv")
val_df = pd.read_csv("data/splits_multilingual/internal_val.csv")
test_df = pd.read_csv("data/splits_multilingual/test.csv")

def get_group(idx): return '_'.join(idx.split('_')[:2])
train_pool = pd.concat([train_df, val_df])
train_pool['group'] = train_pool['id'].apply(get_group)
test_df['group'] = test_df['id'].apply(get_group)

for lang in ["Tamil", "English", "Hindi"]:
    print(f"\n--- {lang} ---")
    tr = train_pool[train_pool['language'] == lang]
    te = test_df[test_df['language'] == lang]
    print("Train groups:", tr['group'].unique())
    print("Test groups:", te['group'].unique())
