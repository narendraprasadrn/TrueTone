import sys
import duckdb
import json

i = int(sys.argv[1])
url = f'https://huggingface.co/datasets/SherryT997/IndicTTS-Deepfake-Challenge-Data/resolve/main/data/train-{i:05d}-of-00035.parquet'

try:
    query = f"""
    SELECT 
        count(*) as rows, 
        sum(CASE WHEN is_tts = 0 THEN 1 ELSE 0 END) as real, 
        sum(CASE WHEN is_tts = 1 THEN 1 ELSE 0 END) as synthetic, 
        list(distinct language) as langs, 
        list(distinct split_part(id, '_', 1) || '_' || split_part(id, '_', 2) || '_' || split_part(id, '_', 3)) as id_prefixes 
    FROM read_parquet('{url}')
    """
    df = duckdb.query(query).df()
    row = df.iloc[0]
    res = {
        'shard': i,
        'rows': int(row['rows']),
        'real': int(row['real']),
        'synthetic': int(row['synthetic']),
        'languages': list(row['langs']),
        'prefixes': list(row['id_prefixes'])
    }
    with open(f'shard_{i:02d}.json', 'w') as f:
        json.dump(res, f)
except Exception as e:
    print(f"Error {i}: {e}")
