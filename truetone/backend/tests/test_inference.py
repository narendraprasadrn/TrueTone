import pytest
import os
import numpy as np
import librosa
from app.preprocessing.pipeline import preprocess_for_detection, prepare_aasist_context
from app.detection.aasist import AasistDetector
from app.detection.prosody import ProsodyDetector
from app.risk_engine.fusion import RiskEngine
import torch

base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
aasist = AasistDetector(os.path.join(base_dir, "third_party/aasist/models/weights/AASIST-L.pth"))
prosody = ProsodyDetector(os.path.join(base_dir, "app/detection/models/prosody_lgbm.txt"))
re = RiskEngine(os.path.join(base_dir, "app/risk_engine/config.yaml"))

def test_aasist_context_preparation():
    # 2 seconds of audio
    sr = 16000
    audio = np.random.randn(sr * 2).astype(np.float32)
    context = prepare_aasist_context(audio)
    
    assert len(context) == 64600
    # Should be zero-padded at the beginning
    assert np.all(context[:32600] == 0.0)
    assert np.array_equal(context[-32000:], audio)
    
def test_aasist_inference():
    # Provide a proper contiguous context
    y, sr = librosa.load(os.path.join(os.path.dirname(base_dir), "test_libri.wav"), sr=None)
    y_proc = preprocess_for_detection(y, sr)
    context = prepare_aasist_context(y_proc)
    
    res = aasist.predict_aasist(context, 16000)
    
    # 8. expected shape of logits
    assert len(res["logits"]) == 2
    
    # 9. softmax scores sum approximately to 1
    assert abs((res["spoof_score"] + res["bonafide_score"]) - 1.0) < 1e-4
    
    # 10. verification
    assert res["spoof_score"] < 0.5  # Real audio should be low spoof score (it was 0.044)

def test_prosody_inference():
    y, sr = librosa.load(os.path.join(base_dir, "synthetic_cloned.ogg"), sr=None)
    y_proc = preprocess_for_detection(y[:2*sr], sr)
    
    res = prosody.score(y_proc, 16000)
    
    # 1. raw output is dict
    assert "bonafide_score" in res
    assert "spoof_score" in res
    
    # 3. spoof_score = 1 - bonafide_score
    assert abs(res["spoof_score"] - (1.0 - res["bonafide_score"])) < 1e-4
    
    # 4. scores remain in [0,1]
    assert 0.0 <= res["spoof_score"] <= 1.0
    assert 0.0 <= res["bonafide_score"] <= 1.0
    
    # 5. real/synthetic semantic direction is correct
    assert res["spoof_score"] > 0.9  # Synthetic audio should have high spoof score

def test_risk_engine():
    res = re.score_window("test_call", 0.75, 0.25, None, {})
    # R_window = 0.75 * 0.75 + 0.25 * 0.25 = 0.5625 + 0.0625 = 0.625
    assert 0.62 < res.r_window < 0.63
    
    assert 0.0 <= res.r_final <= 1.0

if __name__ == "__main__":
    pytest.main(["-v", __file__])
