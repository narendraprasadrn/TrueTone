import pandas as pd
df = pd.read_csv('data/splits_multilingual/train.csv')
test_df = pd.read_csv('data/splits_multilingual/test.csv')
full = pd.concat([df, test_df])
full['group'] = full['id'].apply(lambda x: '_'.join(x.split('_')[:2]))
for lang in ['Tamil', 'English', 'Hindi']:
    l_df = full[full['language'] == lang]
    print(f"{lang}: total={len(l_df)}, real={len(l_df[l_df['is_tts']==0])}, synth={len(l_df[l_df['is_tts']==1])}, groups={l_df['group'].nunique()}")
