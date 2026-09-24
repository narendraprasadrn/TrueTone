import pandas as pd
train_df = pd.read_csv("data/splits_multilingual/train.csv")
en = train_df[train_df['language'] == 'English']
print("English groups in train.csv:", en['group_key'].unique())
