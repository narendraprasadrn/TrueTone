import json
import lightgbm as lgb
import os

models = [
    "models/prosody/v3/lightgbm_tamil_v1.txt",
    "models/prosody/v3/lightgbm_english_v1.txt",
    "models/prosody/v3/lightgbm_hindi_v1.txt",
    "models/prosody/v3/lightgbm_teh_combined_v1.txt"
]

for p in models:
    if os.path.exists(p):
        booster = lgb.Booster(model_file=p)
        print(f"--- {os.path.basename(p)} ---")
        print(f"Num trees: {booster.num_trees()}")
        print(f"Num features: {booster.num_feature()}")
        print(f"Objective: {booster.params.get('objective', 'binary (inferred)')}") # the params might not have objective if loaded from txt
    else:
        print(f"Missing: {p}")

print("--- Data stats ---")
try:
    with open("reports/prosody/v3/teh_combined_results.json", "r") as f:
        teh = json.load(f)
        print(f"Combined Real: {sum(v['real_count'] for k,v in teh['per_language'].items())}")
        print(f"Combined Synth: {sum(v['synthetic_count'] for k,v in teh['per_language'].items())}")
        for k, v in teh['per_language'].items():
            print(f"{k}: {v['sample_count']} total ({v['real_count']} real, {v['synthetic_count']} synth)")
except Exception as e:
    print(e)
