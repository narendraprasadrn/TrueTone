import json
import sys
import logging
from pathlib import Path
from typing import Dict, Any

import numpy as np
import torch
import torch.nn.functional as F

AASIST_DIR = Path(__file__).resolve().parent.parent.parent / "third_party" / "aasist"
if str(AASIST_DIR) not in sys.path:
    sys.path.append(str(AASIST_DIR))

from models.AASIST import Model

# Optional debug flag via environment
import os
AASIST_DEBUG = os.environ.get("AASIST_DEBUG", "false").lower() == "true"

class AasistDetector:
    def __init__(self, checkpoint_path: str, device: str = "cpu"):
        self.device = torch.device(device)
        self.checkpoint_path = checkpoint_path
        
        config_path = AASIST_DIR / "config" / "AASIST-L.conf"
        with open(config_path, "r") as f:
            config = json.load(f)
            
        model_config = config["model_config"]
        
        self.model = Model(model_config).to(self.device)
        
        # Strict load
        sd = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(sd, strict=True)
        self.model.eval()
        
        self.max_len = 64600
        
        if AASIST_DEBUG:
            print(f"AASIST-L checkpoint: {checkpoint_path}")
            print(f"Checkpoint loaded: true")
            print(f"Missing keys: []")
            print(f"Unexpected keys: []")
            print(f"Parameter count: {sum(p.numel() for p in self.model.parameters())}")
            print(f"Model mode: eval")

    def predict_aasist(self, window: np.ndarray, sr: int = 16000) -> Dict[str, Any]:
        """
        class 0 = spoof
        class 1 = bonafide
        """
        if sr != 16000:
            raise ValueError(f"AASIST expects sample rate 16000, got {sr}")
            
        if window.ndim > 1:
            window = window.squeeze()
            
        window = window.astype(np.float32)
        
        assert len(window) == self.max_len, f"AASIST requires exactly {self.max_len} samples, got {len(window)}"
        window_padded = window
        
        # Log properties before processing
        w_min, w_max, w_mean, w_std = window_padded.min(), window_padded.max(), window_padded.mean(), window_padded.std()
            
        x_inp = torch.Tensor(window_padded).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            _, out = self.model(x_inp)
            logits = out
            assert logits.shape[-1] == 2
            
            probabilities = F.softmax(logits, dim=1)
            assert abs(float(probabilities.sum(dim=1).mean()) - 1.0) < 1e-4
            
        spoof_score = float(probabilities[0, 0].cpu().numpy())
        bonafide_score = float(probabilities[0, 1].cpu().numpy())
        
        logit_0 = float(logits[0, 0].cpu().numpy())
        logit_1 = float(logits[0, 1].cpu().numpy())
        
        predicted_class = "bonafide" if bonafide_score > spoof_score else "spoof"
        
        if AASIST_DEBUG:
            print("[AASIST DEBUG]")
            print(f"sample_rate={sr}")
            print(f"channels=1")
            print(f"input_samples={self.max_len}")
            print(f"duration={self.max_len/sr:.4f}")
            print(f"waveform_min={w_min:.4f}")
            print(f"waveform_max={w_max:.4f}")
            print(f"waveform_mean={w_mean:.4f}")
            print(f"waveform_std={w_std:.4f}")
            print(f"logit_spoof={logit_0:.4f}")
            print(f"logit_bonafide={logit_1:.4f}")
            print(f"spoof_score={spoof_score:.4f}")
            print(f"bonafide_score={bonafide_score:.4f}")
            print(f"prediction={predicted_class}")
            
        return {
            "spoof_score": spoof_score,
            "bonafide_score": bonafide_score,
            "logits": [logit_0, logit_1],
            "predicted_class": predicted_class
        }
        
    def score(self, window: np.ndarray, sr: int = 16000) -> float:
        """Legacy wrapper for compatibility until fully removed."""
        res = self.predict_aasist(window, sr)
        return res["spoof_score"]
