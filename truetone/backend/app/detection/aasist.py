import json
import sys
from pathlib import Path
from typing import Dict, Any

import numpy as np
import torch
import torch.nn.functional as F

AASIST_DIR = Path(__file__).resolve().parent.parent.parent / "third_party" / "aasist"
if str(AASIST_DIR) not in sys.path:
    sys.path.append(str(AASIST_DIR))

from models.AASIST import Model

class AasistDetector:
    def __init__(self, checkpoint_path: str, device: str = "cpu"):
        self.device = torch.device(device)
        self.checkpoint_path = checkpoint_path
        self.model_version = Path(checkpoint_path).name
        
        config_path = AASIST_DIR / "config" / "AASIST-L.conf"
        with open(config_path, "r") as f:
            config = json.load(f)
            
        model_config = config["model_config"]
        
        self.model = Model(model_config).to(self.device)
        self.model.load_state_dict(torch.load(checkpoint_path, map_location=self.device))
        self.model.eval()
        
        self.max_len = 64600

    def pad(self, x: np.ndarray) -> np.ndarray:
        x_len = x.shape[0]
        if x_len >= self.max_len:
            return x[:self.max_len]
        num_repeats = int(self.max_len / x_len) + 1
        padded_x = np.tile(x, (num_repeats,))[:self.max_len]
        return padded_x

    def score(self, window: np.ndarray, sr: int = 16000) -> float:
        if sr != 16000:
            raise ValueError(f"AASIST expects sample rate 16000, got {sr}")
            
        if window.ndim > 1:
            window = window.squeeze()
            
        window = window.astype(np.float32)
        window = self.pad(window)
            
        x_inp = torch.Tensor(window).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            _, out = self.model(x_inp)
            probs = F.softmax(out, dim=1)
            
        spoof_prob = float(probs[0, 0].cpu().numpy())
        return spoof_prob
