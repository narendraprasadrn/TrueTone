import os
from pathlib import Path
import numpy as np
import lightgbm as lgb
from app.detection.prosody_features import extract_prosody_features

class ProsodyDetector:
    def __init__(self, model_path: str = None):
        if model_path is None:
            model_path = str(Path(__file__).resolve().parent / "models" / "prosody_lgbm.txt")
        self.model_path = model_path
        self.model = None
        if os.path.exists(self.model_path):
            self.model = lgb.Booster(model_file=self.model_path)
            
    def score(self, window: np.ndarray, sr: int = 16000) -> float:
        """Returns behavioral-anomaly probability 0-1."""
        features = extract_prosody_features(window, sr)
        
        if self.model is not None:
            # Reshape to (1, num_features)
            features = features.reshape(1, -1)
            
            # Predict returns probability for binary classification
            # The model was trained with 1 = bonafide, 0 = spoof.
            # We want to return the spoof probability, so we return 1.0 - prob
            prob_bonafide = self.model.predict(features)[0]
            prob_bonafide = float(np.clip(prob_bonafide, 0.0, 1.0))
            return {
                "bonafide_score": prob_bonafide,
                "spoof_score": 1.0 - prob_bonafide
            }
            
        # PLACEHOLDER — rule-based, to be replaced by trained LightGBM once a
        # labeled dataset (ASVspoof LA or bootstrap real+TTS set) is available.
        # Features unpacked from extract_prosody_features (length 10)
        f0_mean, f0_std, f0_range, jitter, shimmer, voiced_ratio, pause_count, pause_mean_dur, flatness, hnr = features
        
        print(f"[ProsodyDetector] RAW FEATURES: f0_mean={f0_mean:.2f}, f0_std={f0_std:.2f}, f0_range={f0_range:.2f}, jitter={jitter:.4f}, shimmer={shimmer:.4f}, voiced_ratio={voiced_ratio:.2f}, pause_count={pause_count}, pause_mean_dur={pause_mean_dur:.2f}, flatness={flatness:.6f}, hnr={hnr:.2f}")
        
        anomaly_score = 0.0
        
        # 1. Pitch variation (TTS often lacks dynamic pitch contour)
        if f0_std < 20.0:
            anomaly_score += 0.3 * (20.0 - f0_std) / 20.0
            
        # 2. Jitter and Shimmer (TTS can be too "perfect")
        if jitter < 0.01:
            anomaly_score += 0.2 * (0.01 - jitter) / 0.01
        elif jitter > 0.04:
            anomaly_score += 0.2 * min(1.0, (jitter - 0.04) / 0.04)
            
        if shimmer < 0.05:
            anomaly_score += 0.2 * (0.05 - shimmer) / 0.05
        elif shimmer > 0.12:
            anomaly_score += 0.2 * min(1.0, (shimmer - 0.12) / 0.12)
            
        # 3. Speaking rate / Voiced ratio (TTS might lack natural pauses)
        if voiced_ratio > 0.80:
            anomaly_score += 0.2 * min(1.0, (voiced_ratio - 0.80) / 0.20)
            
        # 4. Harmonic-to-Noise Ratio (TTS can be excessively harmonic)
        if hnr > 20.0:
            anomaly_score += 0.2 * min(1.0, (hnr - 20.0) / 10.0)
            
        # 5. Spectral flatness (TTS often lacks natural high-frequency breath noise)
        if flatness < 0.001:
            anomaly_score += 0.1 * (0.001 - flatness) / 0.001
            
        return {
            "bonafide_score": 1.0 - min(1.0, float(anomaly_score)),
            "spoof_score": min(1.0, float(anomaly_score))
        }
