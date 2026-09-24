import pandas as pd
train_df = pd.read_csv("data/splits_multilingual/internal_train.csv")
val_df = pd.read_csv("data/splits_multilingual/internal_val.csv")
train_pool = pd.concat([train_df, val_df])

eng = train_pool[train_pool['language'] == 'English']
print("English IDs:")
print(eng['id'].head(10).tolist())

hin = train_pool[train_pool['language'] == 'Hindi']
print("Hindi IDs:")
print(hin['id'].head(10).tolist())
