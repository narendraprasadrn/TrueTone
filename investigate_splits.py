import pandas as pd
train_df = pd.read_csv("data/splits_multilingual/train.csv")
val_df = pd.read_csv("data/splits_multilingual/val.csv")
test_df = pd.read_csv("data/splits_multilingual/test.csv")
print("train.csv groups:", train_df['group_key'].unique())
print("val.csv groups:", val_df['group_key'].unique())
