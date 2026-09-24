import pandas as pd
train_df = pd.read_csv("data/splits_multilingual/internal_train.csv")
val_df = pd.read_csv("data/splits_multilingual/internal_val.csv")
test_df = pd.read_csv("data/splits_multilingual/test.csv")

pool = pd.concat([train_df, val_df])

for lang in ["Tamil", "English", "Hindi"]:
    subset = pool[pool['language'] == lang]
    print(f"{lang} groups:", subset['group_key'].unique())
    
    test_sub = test_df[test_df['language'] == lang]
    print(f"{lang} Test groups:", test_sub['group_key'].unique())
