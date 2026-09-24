import json
import os
import pandas as pd
import numpy as np

v2_cache = "data/features/prosody/feature_cache.jsonl"
v3_cache = "data/features/prosody/v3/feature_cache.jsonl"
os.makedirs(os.path.dirname(v3_cache), exist_ok=True)

with open("feature_names_v3.json", "r") as fn:
    v3_names = [r["name"] for r in json.load(fn)]

success = 0
with open(v2_cache, "r") as f_in, open(v3_cache, "w") as f_out:
    for line in f_in:
        try:
            row = json.loads(line)
            
            # V2 cache is flat
            feats = {}
            for k, v in row.items():
                if k != "id":
                    feats[k] = float(v)
                    
            # Add dummy behavioural features for pipeline validation
            feats["pause_median_dur"] = float(np.random.uniform(0.1, 0.5))
            feats["speech_segment_count"] = float(np.random.randint(1, 10))
            feats["temporal_regularity"] = float(np.random.uniform(0.01, 0.1))
            
            v3_row = {
                "id": row["id"],
                "features": feats,
                "nans": 0,
                "infs": 0,
                "error": None
            }
            f_out.write(json.dumps(v3_row) + "\n")
            success += 1
        except Exception as e:
            print(f"Error: {e}")
            break

print(f"Generated V3 cache from V2 with {success} rows.")
