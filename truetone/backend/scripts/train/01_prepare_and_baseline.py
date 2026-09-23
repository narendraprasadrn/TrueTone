import os
import json
import pandas as pd
import numpy as np
from datasets import load_dataset
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
from app.preprocessing.pipeline import preprocess_for_detection, prepare_aasist_context
from app.detection.aasist import AasistDetector

def compute_eer(y_true, y_score):
    from sklearn.metrics import roc_curve
    fpr, tpr, thresholds = roc_curve(y_true, y_score)
    fnr = 1 - tpr
    eer = fpr[np.nanargmin(np.absolute((fnr - fpr)))]
    return eer

def main():
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    os.makedirs(os.path.join(base_dir, "data/splits"), exist_ok=True)
    os.makedirs(os.path.join(base_dir, "reports"), exist_ok=True)
    
    print("Downloading/Loading full dataset (this will take a while)...")
    ds = load_dataset("SherryT997/IndicTTS-Deepfake-Challenge-Data", "default")
    
    train_ds = ds["train"]
    print("Dataset loaded successfully!")
    
    # Extract metadata
    ids = []
    langs = []
    labels = []
    
    # We can use train_ds directly since it's cached locally now
    for i in range(len(train_ds)):
        ids.append(train_ds[i]["id"])
        langs.append(train_ds[i]["language"])
        labels.append(train_ds[i]["is_tts"])
        
    df = pd.DataFrame({"id": ids, "language": langs, "is_tts": labels, "index": range(len(train_ds))})
    
    # Determine speaker/source grouping.
    # We assume the ID prefix (e.g. before the last underscore or some pattern) identifies the speaker/source
    # E.g., 'ASM_F_ANGER_00109' -> 'ASM_F_ANGER' could be the speaker + emotion
    df["group"] = df["id"].apply(lambda x: "_".join(x.split("_")[:-1]) if "_" in x else x)
    
    # Check duplicates
    dup_ids = df["id"].duplicated().sum()
    print(f"Duplicate IDs: {dup_ids}")
    
    # Split using GroupShuffleSplit to ensure no group overlaps
    gss = GroupShuffleSplit(n_splits=1, train_size=0.8, random_state=42)
    train_idx, temp_idx = next(gss.split(df, groups=df["group"]))
    
    df_train = df.iloc[train_idx]
    df_temp = df.iloc[temp_idx]
    
    gss2 = GroupShuffleSplit(n_splits=1, train_size=0.5, random_state=42)
    val_idx, test_idx = next(gss2.split(df_temp, groups=df_temp["group"]))
    
    df_val = df_temp.iloc[val_idx]
    df_test = df_temp.iloc[test_idx]
    
    df_train.to_csv(os.path.join(base_dir, "data/splits/train.csv"), index=False)
    df_val.to_csv(os.path.join(base_dir, "data/splits/validation.csv"), index=False)
    df_test.to_csv(os.path.join(base_dir, "data/splits/test.csv"), index=False)
    
    print("Splits saved.")
    
    # BASELINE ON TEST SPLIT
    print("Running baseline on our test split...")
    aasist = AasistDetector(os.path.join(base_dir, "third_party/aasist/models/weights/AASIST-L.pth"))
    
    y_true = []
    y_scores = []
    y_pred = []
    
    for _, row in df_test.iterrows():
        idx = row["index"]
        ex = train_ds[idx]
        
        audio = ex["audio"]["array"]
        sr = ex["audio"]["sampling_rate"]
        label = ex["is_tts"]
        
        y_proc = preprocess_for_detection(audio, sr)
        context = prepare_aasist_context(y_proc[:64600])
        
        res = aasist.predict_aasist(context, 16000)
        spoof_score = res["spoof_score"]
        
        y_true.append(label)
        y_scores.append(spoof_score)
        y_pred.append(1 if spoof_score > 0.5 else 0)
        
    y_true = np.array(y_true)
    y_scores = np.array(y_scores)
    y_pred = np.array(y_pred)
    
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    roc = roc_auc_score(y_true, y_scores)
    eer = compute_eer(y_true, y_scores)
    
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0,1]).ravel()
    fpr = fp / (fp + tn) if (fp+tn) > 0 else 0
    fnr = fn / (fn + tp) if (fn+tp) > 0 else 0
    
    report = {
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1": f1,
        "roc_auc": roc,
        "eer": eer,
        "fpr": fpr,
        "fnr": fnr
    }
    
    with open(os.path.join(base_dir, "reports/aasist_pretrained_baseline.json"), "w") as f:
        json.dump(report, f, indent=2)
        
    print("Baseline report saved.")

if __name__ == "__main__":
    main()
