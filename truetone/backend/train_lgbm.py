import os
import sys
import glob
from pathlib import Path

import numpy as np
import soundfile as sf
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

from app.detection.prosody_features import extract_prosody_features

def main():
    if len(sys.argv) < 2:
        print("Usage: python train_lgbm.py <path_to_dataset>")
        print("Dataset should have 'bonafide' and 'spoof' subdirectories containing .wav or .flac files.")
        sys.exit(1)
        
    dataset_path = Path(sys.argv[1])
    bonafide_dir = dataset_path / "bonafide"
    spoof_dir = dataset_path / "spoof"
    
    if not bonafide_dir.exists() or not spoof_dir.exists():
        print(f"Error: Could not find 'bonafide' or 'spoof' directories in {dataset_path}")
        sys.exit(1)
        
    X = []
    y = []
    
    print("Extracting features for bonafide clips...")
    for ext in ["*.wav", "*.flac"]:
        for file in bonafide_dir.glob(ext):
            audio, sr = sf.read(str(file))
            feat = extract_prosody_features(audio, sr)
            X.append(feat)
            y.append(0) # 0 = genuine
            
    print("Extracting features for spoof clips...")
    for ext in ["*.wav", "*.flac"]:
        for file in spoof_dir.glob(ext):
            audio, sr = sf.read(str(file))
            feat = extract_prosody_features(audio, sr)
            X.append(feat)
            y.append(1) # 1 = spoof
            
    X = np.array(X)
    y = np.array(y)
    
    if len(X) == 0:
        print("Error: No audio files found.")
        sys.exit(1)
        
    print(f"Extracted features for {len(X)} files. Training LightGBM...")
    
    # Simple train/test split if enough data, else just train on all
    if len(X) > 10:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    else:
        X_train, X_test, y_train, y_test = X, X, y, y
        
    train_data = lgb.Dataset(X_train, label=y_train)
    test_data = lgb.Dataset(X_test, label=y_test, reference=train_data)
    
    params = {
        'objective': 'binary',
        'metric': 'binary_error',
        'boosting_type': 'gbdt',
        'learning_rate': 0.1,
        'num_leaves': 15,
        'max_depth': 5,
        'verbose': -1
    }
    
    model = lgb.train(
        params,
        train_data,
        num_boost_round=100,
        valid_sets=[test_data],
        callbacks=[lgb.early_stopping(stopping_rounds=10)]
    )
    
    preds = model.predict(X_test)
    pred_labels = (preds > 0.5).astype(int)
    acc = accuracy_score(y_test, pred_labels)
    
    print(f"Validation Accuracy: {acc:.4f}")
    
    models_dir = Path(__file__).resolve().parent / "app" / "detection" / "models"
    models_dir.mkdir(parents=True, exist_ok=True)
    
    model_path = models_dir / "prosody_lgbm.txt"
    model.save_model(str(model_path))
    
    print(f"Model saved to {model_path}")

if __name__ == "__main__":
    main()
