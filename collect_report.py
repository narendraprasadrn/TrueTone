import json
import pandas as pd
import os

print("--- BASELINE SUMMARY ---")
with open("reports/baseline/aasist_l_baseline_summary.json") as f: print(f.read())

print("\n--- FINETUNING CONFIG ---")
if os.path.exists("config/train_aasist_tamil_hindi.yaml"):
    with open("config/train_aasist_tamil_hindi.yaml") as f: print(f.read())

print("\n--- MULTILINGUAL METRICS (FINE-TUNED) ---")
if os.path.exists("reports/finetuning_tamil_hindi/multilingual_metrics.json"):
    with open("reports/finetuning_tamil_hindi/multilingual_metrics.json") as f: print(f.read())

print("\n--- COMPARISON (PRETRAINED VS FINETUNED) ---")
if os.path.exists("reports/finetuning_tamil_hindi/pretrained_vs_finetuned.csv"):
    df = pd.read_csv("reports/finetuning_tamil_hindi/pretrained_vs_finetuned.csv")
    print(df.to_string())

print("\n--- WHATSAPP (BASELINE) ---")
if os.path.exists("reports/baseline/whatsapp_external_check.json"):
    with open("reports/baseline/whatsapp_external_check.json") as f: print(f.read())

print("\n--- WHATSAPP (FINE-TUNED) ---")
if os.path.exists("reports/finetuning_tamil_hindi/whatsapp_external_check.json"):
    with open("reports/finetuning_tamil_hindi/whatsapp_external_check.json") as f: print(f.read())

