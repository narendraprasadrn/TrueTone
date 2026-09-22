import torch
ckpt = torch.load("backend/third_party/aasist/models/weights/AASIST-L.pth", map_location="cpu")
print("Type of checkpoint:", type(ckpt))
if isinstance(ckpt, dict):
    print("Keys in checkpoint:", list(ckpt.keys())[:10])
