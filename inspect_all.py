import duckdb
import concurrent.futures
import json
import time

urls = [f'https://huggingface.co/datasets/SherryT997/IndicTTS-Deepfake-Challenge-Data/resolve/main/data/train-{i:05d}-of-00035.parquet' for i in range(35)]

def inspect_shard(i):
    url = urls[i]
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
        return {
            'shard': i,
            'rows': int(row['rows']),
            'real': int(row['real']),
            'synthetic': int(row['synthetic']),
            'languages': list(row['langs']),
            'prefixes': list(row['id_prefixes'])
        }
    except Exception as e:
        print(f"Error on {i}: {e}")
        return None

results = []
start = time.time()
with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
    futures = {executor.submit(inspect_shard, i): i for i in range(35)}
    for future in concurrent.futures.as_completed(futures):
        res = future.result()
        if res:
            results.append(res)
            print(f"Finished shard {res['shard']}")

results.sort(key=lambda x: x['shard'])

with open('all_shards.json', 'w') as f:
    json.dump(results, f, indent=2)

print(f"Done in {time.time() - start:.1f}s")
