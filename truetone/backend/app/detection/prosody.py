import os
import numpy as np
import librosa
from typing import Dict, Any

PROSODY_ORDER = ["f0_mean", "f0_std_st", "f0_range_st", "voiced_frac", "jitter", "shimmer",
                 "energy_cv", "pause_ratio", "pause_rate", "flatness", "centroid_cv", "mfcc_std"]

def prosody_features(x: np.ndarray, sr: int = 16000) -> Dict[str, float]:
    hop, frame = 256, 1024
    f0, voiced, _ = librosa.pyin(x, fmin=librosa.note_to_hz("C2"), fmax=librosa.note_to_hz("C6"),
                                 sr=sr, frame_length=frame, hop_length=hop)
    rms = librosa.feature.rms(y=x, frame_length=frame, hop_length=hop)[0]
    n = min(len(f0), len(rms))
    f0, voiced, rms = f0[:n], voiced[:n] & ~np.isnan(f0[:n]), rms[:n]
    idx = np.where(voiced)[0]
    feats = dict.fromkeys(PROSODY_ORDER, 0.0)
    feats["voiced_frac"] = float(voiced.mean()) if n else 0.0

    if len(idx) >= 10:
        f0v = f0[idx]
        semis = 12 * np.log2(f0v / f0v.mean())
        pair = idx[1:] - idx[:-1] == 1
        d_f0 = np.abs(np.diff(f0v))[pair]
        d_rms = np.abs(np.diff(rms[idx]))[pair]
        feats.update(
            f0_mean=float(f0v.mean()),
            f0_std_st=float(semis.std()),
            f0_range_st=float(np.percentile(semis, 95) - np.percentile(semis, 5)),
            jitter=float(d_f0.mean() / f0v.mean()) if len(d_f0) else 0.0,
            shimmer=float(d_rms.mean() / rms[idx].mean()) if len(d_rms) else 0.0,
        )
    feats["energy_cv"] = float(rms.std() / (rms.mean() + 1e-8))

    db = 20 * np.log10(rms + 1e-8)
    silent = db < (db.max() - 40)
    feats["pause_ratio"] = float(silent.mean())
    runs, run = 0, 0
    for s in silent:
        run = run + 1 if s else 0
        if run == 10:
            runs += 1
    feats["pause_rate"] = float(runs / (len(x) / sr))

    feats["flatness"] = float(librosa.feature.spectral_flatness(y=x).mean())
    cen = librosa.feature.spectral_centroid(y=x, sr=sr)[0]
    feats["centroid_cv"] = float(cen.std() / (cen.mean() + 1e-8))
    feats["mfcc_std"] = float(librosa.feature.mfcc(y=x, sr=sr, n_mfcc=13).std(axis=1).mean())
    return feats


def _sat(v, lo, hi):
    return float(np.clip((v - lo) / (hi - lo), 0, 1))


def heuristic_prosody_real(f):
    if f["voiced_frac"] < 0.1:
        return 0.5
    return float(np.mean([_sat(f["f0_std_st"], 0.5, 3.0), _sat(f["jitter"], 0.002, 0.02),
                          _sat(f["shimmer"], 0.02, 0.15), _sat(f["energy_cv"], 0.3, 1.0)]))

class ProsodyDetector:
    def __init__(self, model_path: str = None):
        pass # Streamlit ignores backend lgbm model and uses heuristic logic directly for now

    def score(self, window: np.ndarray, sr: int = 16000) -> Dict[str, float]:
        """Returns behavioral-anomaly probability 0-1 as a dictionary."""
        feats = prosody_features(window, sr)
        prob_bonafide = heuristic_prosody_real(feats)
        
        return {
            "bonafide_score": prob_bonafide,
            "spoof_score": 1.0 - prob_bonafide
        }
