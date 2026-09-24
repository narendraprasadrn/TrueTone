import pandas as pd
val_df = pd.read_csv("data/splits_multilingual/internal_val.csv")
print("English in internal_val.csv:", len(val_df[val_df['language'] == 'English']))
