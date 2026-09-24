import json
import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import GroupKFold
import warnings
warnings.filterwarnings('ignore')

manifest_dfs = [pd.read_csv(f) for f in ["data/splits_multilingual/internal_train.csv", "data/splits_multilingual/internal_val.csv", "data/splits_multilingual/test.csv"]]
all_manifest = pd.concat(manifest_dfs).set_index('id')
with open("feature_names_v3.json", "r") as fn:
    v3_info = json.load(fn)
v3_names = [r["name"] for r in v3_info]

data = []
with open("data/features/prosody/v3/feature_cache.jsonl", "r") as f:
    for line in f:
        row = json.loads(line)
        if row["id"] in all_manifest.index:
            manifest_row = all_manifest.loc[row["id"]]
            if manifest_row["language"] == "English":
                vec = [row["features"].get(n, 0.0) for n in v3_names]
                data.append({
                    "id": row["id"],
                    "is_tts": manifest_row["is_tts"],
                    "group": '_'.join(row["id"].split('_')[:2]),
                    "features": vec
                })

df = pd.DataFrame(data)
X = np.array(df["features"].tolist())
y = df["is_tts"].values
groups = df["group"].values

cv = GroupKFold(n_splits=2)
for train_idx, test_idx in cv.split(X, y, groups):
    X_tr, y_tr = X[train_idx], y[train_idx]
    X_val, y_val = X[test_idx], y[test_idx]
    
    print("Train group:", np.unique(groups[train_idx]))
    print("Val group:", np.unique(groups[test_idx]))
    
    model = lgb.LGBMClassifier(objective='binary', learning_rate=0.03, num_leaves=31, max_depth=-1, min_child_samples=20, n_estimators=1000, verbose=-1)
    
    model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], callbacks=[lgb.early_stopping(50, verbose=True)])
    print("Best iteration:", model.best_iteration_)
    
    # wait let me just run it for both folds
