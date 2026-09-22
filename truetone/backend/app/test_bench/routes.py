import os
import shutil
import tempfile
import uuid
from typing import Optional
from fastapi import APIRouter, UploadFile, File, Form, HTTPException
import librosa
import numpy as np

from app.preprocessing.pipeline import preprocess_for_detection
from app.detection.aasist import AasistDetector
from app.detection.prosody import ProsodyDetector
from app.detection.speaker_verification import SpeakerVerifier
from app.risk_engine.fusion import RiskEngine

router = APIRouter(prefix="/test", tags=["test_bench"])

# Global models to avoid reloading on every upload (lazy loaded)
_aasist = None
_prosody = None
_sv = None
_re = None

def get_models():
    global _aasist, _prosody, _sv, _re
    # __file__ is backend/app/test_bench/routes.py
    # so base_dir should be backend/
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    
    if _aasist is None:
        _aasist = AasistDetector(os.path.join(base_dir, "third_party/aasist/models/weights/AASIST-L.pth"))
    if _prosody is None:
        _prosody = ProsodyDetector(os.path.join(base_dir, "app/detection/models/prosody_lgbm.txt")) 
    if _sv is None:
        _sv = SpeakerVerifier(savedir=os.path.join(base_dir, "pretrained_models/ecapa-tdnn"))
    if _re is None:
        _re = RiskEngine(os.path.join(base_dir, "app/risk_engine/config.yaml"))
    return _aasist, _prosody, _sv, _re

@router.post("/analyze-audio")
async def analyze_audio(
    file: UploadFile = File(...),
    enrolled_identity_id: Optional[str] = Form(None)
):
    # Reject files over 20MB
    file.file.seek(0, 2) # seek to end
    file_size = file.file.tell()
    file.file.seek(0) # reset
    
    if file_size > 20 * 1024 * 1024:
        raise HTTPException(status_code=400, detail="File too large. Maximum size is 20MB.")
        
    ext = os.path.splitext(file.filename)[1].lower()
    if ext not in [".wav", ".mp3", ".m4a", ".ogg", ".flac"]:
        raise HTTPException(status_code=400, detail=f"Unsupported file format: {ext}")
        
    temp_fd, temp_path = tempfile.mkstemp(suffix=ext)
    os.close(temp_fd)
    
    try:
        # Save upload to temp file
        with open(temp_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
            
        # Load audio using librosa (handles resampling natively)
        try:
            audio_data, sr = librosa.load(temp_path, sr=None) # Keep original sr for preprocessing
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Failed to decode audio: {str(e)}")
            
        # Duration check
        duration_s = librosa.get_duration(y=audio_data, sr=sr)
        if duration_s > 120.0:
            raise HTTPException(status_code=400, detail="Audio exceeds maximum duration of 2 minutes.")
            
        if len(audio_data.shape) > 1:
            audio_data = audio_data[0] # mono
            
        # Load models
        aasist, prosody, sv, re = get_models()
            
        # 3-5s rolling windows (using 3s windows, 1s step as in CallSimulator)
        window_duration_sec = 2.0
        step_duration_sec = 1.0
        
        window_samples = int(window_duration_sec * sr)
        step_samples = int(step_duration_sec * sr)
        
        total_samples = len(audio_data)
        start_idx = 0
        
        windows_result = []
        
        call_id = f"test_{uuid.uuid4().hex[:8]}"
        
        
        window_index = 0
        while start_idx + window_samples <= total_samples:
            end_idx = start_idx + window_samples
            window_chunk = audio_data[start_idx:end_idx]
            
            # Preprocess
            processed = preprocess_for_detection(window_chunk, sr, debug=False)
            
            # Detect
            a_score = aasist.score(processed, 16000)
            p_score = prosody.score(processed, 16000)
            s_score = sv.score(processed, enrolled_identity_id) if enrolled_identity_id else None
            
            # Fuse
            ctx_flags = {"unknown_caller": False, "high_value_keywords": False, "ivr_allowlisted": False}
            win_res = re.score_window(call_id, a_score, p_score, s_score, ctx_flags)
            
            windows_result.append({
                "window_index": window_index,
                "start_s": start_idx / sr,
                "end_s": end_idx / sr,
                "aasist_score": a_score,
                "prosody_score": p_score,
                "speaker_score": s_score,
                "fused_score": win_res.r_final,
                "classification": win_res.classification
            })
            
            start_idx += step_samples
            window_index += 1
            
        # Final status
        final_state = re.call_states.get(call_id)
        if final_state and final_state.history:
            overall_score = final_state.history[-1]
            overall_class = final_state.classification
        else:
            overall_score = 0.0
            overall_class = "LOW" 
        
        # Cleanup state
        if call_id in re.call_states:
            del re.call_states[call_id]
            
        return {
            "filename": file.filename,
            "duration_seconds": duration_s,
            "windows": windows_result,
            "overall_score": overall_score,
            "overall_classification": overall_class
        }
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)
