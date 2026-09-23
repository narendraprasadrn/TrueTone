import os
import sys
import json
import io
import time
import numpy as np
import lightgbm as lgb
import soundfile as sf
import datasets
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix

base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, base_dir)

from detection.prosody_features import extract_prosody_features

def main():
    print("Loading ASVspoof 2019 LA from Hugging Face (streaming)...")
    
    # We need to collect:
    # Train: 1000 bonafide, 1000 spoof
    # Val: 250 bonafide, 250 spoof
    # Test: 250 bonafide, 250 spoof
    
    # Since only the 'test' split is available on HF, we will partition it.
    ds = datasets.load_dataset("SpeechAntiSpoofingBenchmarks/ASVspoof2019_LA", split="test", streaming=True)
    ds = ds.cast_column("audio", datasets.Audio(decode=False))
    
    counts = {"bonafide": 0, "spoof": 0}
    
    data = {"train": {"X": [], "y": []}, "val": {"X": [], "y": []}, "test": {"X": [], "y": []}}
    
    for ex in ds:
        label = ex.get("label")
        # In ASVspoof LA HF test split, we discovered label=1 is spoof, label=0 is bonafide
        is_spoof = (label == 1 or label == "spoof")
        
        c_type = "spoof" if is_spoof else "bonafide"
        
        # Decide which split this sample goes into based on current count
        c = counts[c_type]
        if c < 1000:
            target_split = "train"
        elif c < 1250:
            target_split = "val"
        elif c < 1500:
            target_split = "test"
        else:
            continue
            
        try:
            audio_bytes = ex["audio"]["bytes"]
            audio_array, sr = sf.read(io.BytesIO(audio_bytes))
            
            if len(audio_array.shape) > 1:
                audio_array = audio_array[:, 0]
            
            if len(audio_array) < 16000:
                continue
                
            features = extract_prosody_features(audio_array, sr)
            
            data[target_split]["X"].append(features)
            data[target_split]["y"].append(1 if is_spoof else 0)
            
            counts[c_type] += 1
            
            total_b = counts["bonafide"]
            total_s = counts["spoof"]
            if (total_b + total_s) % 100 == 0:
                print(f"Loaded {total_b} bonafide, {total_s} spoof...")
                
            if total_b >= 1500 and total_s >= 1500:
                break
        except Exception as e:
            pass
            
    print("\nTraining LightGBM model...")
    X_train = np.array(data["train"]["X"])
    y_train = np.array(data["train"]["y"])
    X_val = np.array(data["val"]["X"])
    y_val = np.array(data["val"]["y"])
    X_test = np.array(data["test"]["X"])
    y_test = np.array(data["test"]["y"])
    
    train_data = lgb.Dataset(X_train, label=y_train)
    val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)
    
    params = {
        "objective": "binary",
        "metric": "binary_logloss",
        "boosting_type": "gbdt",
        "learning_rate": 0.05,
        "num_leaves": 31,
        "verbose": -1
    }
    
    model = lgb.train(
        params,
        train_data,
        valid_sets=[train_data, val_data],
        num_boost_round=200,
        callbacks=[lgb.early_stopping(stopping_rounds=20, verbose=False)]
    )
    
    print("\nEvaluating on Test Set...")
    preds_prob = model.predict(X_test)
    preds_class = (preds_prob > 0.5).astype(int)
    
    metrics = {
        "accuracy": float(accuracy_score(y_test, preds_class)),
        "precision": float(precision_score(y_test, preds_class)),
        "recall": float(recall_score(y_test, preds_class)),
        "f1": float(f1_score(y_test, preds_class)),
        "roc_auc": float(roc_auc_score(y_test, preds_prob))
    }
    
    cm = confusion_matrix(y_test, preds_class)
    fpr = cm[0][1] / (cm[0][0] + cm[0][1]) if (cm[0][0] + cm[0][1]) > 0 else 0.0
    metrics["fpr_on_bonafide"] = float(fpr)
    metrics["confusion_matrix"] = cm.tolist()
    
    print(json.dumps(metrics, indent=2))
    
    models_dir = os.path.dirname(os.path.abspath(__file__)) + "/models"
    os.makedirs(models_dir, exist_ok=True)
    
    model_path = os.path.join(models_dir, "prosody_lgbm.txt")
    model.save_model(model_path)
    print(f"\nModel saved to {model_path}")
    
    with open(os.path.join(models_dir, "training_metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)

if __name__ == "__main__":
    main()
