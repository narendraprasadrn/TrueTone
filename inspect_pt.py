import torch
import sys

ckpt_path = "models/aasist/finetuned_tamil_hindi/best.pt"
print(f"Loading {ckpt_path}")
try:
    ckpt = torch.load(ckpt_path, map_location="cpu")
    print("Keys in checkpoint:", ckpt.keys())
    if "epoch" in ckpt:
        print("Epoch:", ckpt["epoch"])
    if "metrics" in ckpt:
        print("Metrics:", ckpt["metrics"])
    if "optimizer_state_dict" in ckpt:
        print("Optimizer state available.")
    else:
        print("Optimizer state NOT available.")
    if "val_acc" in ckpt:
        print("Val Acc:", ckpt["val_acc"])
    if "config" in ckpt:
        print("Config:", ckpt["config"])
except Exception as e:
    print("Error:", e)
