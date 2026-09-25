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

_aasist = None
_prosody = None
_sv = None
_re = None

def get_models():
    global _aasist, _prosody, _sv, _re
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
    file.file.seek(0, 2)
    file_size = file.file.tell()
    file.file.seek(0)
    if file_size > 20 * 1024 * 1024:
        raise HTTPException(status_code=400, detail="File too large. Maximum size is 20MB.")
        
    ext = os.path.splitext(file.filename)[1].lower()
    temp_fd, temp_path = tempfile.mkstemp(suffix=ext)
    os.close(temp_fd)
    
    try:
        with open(temp_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
            
        try:
            audio_data, sr = librosa.load(temp_path, sr=None)
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Failed to decode audio: {str(e)}")
            
        duration_s = librosa.get_duration(y=audio_data, sr=sr)
        if duration_s > 120.0:
            raise HTTPException(status_code=400, detail="Audio exceeds maximum duration of 2 minutes.")
            
        if len(audio_data.shape) > 1:
            audio_data = audio_data[0]
            
        aasist, prosody, sv, re = get_models()
        
        # We step every 1.0 seconds. 
        # For AASIST, we use 4.0375s history ending at current timestamp.
        # For Prosody/SV, we can use a 2.0s history ending at current timestamp.
        
        step_duration_sec = 1.0
        aasist_window_sec = 64600 / 16000 # 4.0375
        other_window_sec = 2.0
        
        step_samples = int(step_duration_sec * sr)
        total_samples = len(audio_data)
        
        windows_result = []
        call_id = f"test_{uuid.uuid4().hex[:8]}"
        
        # Start at 2.0s to have at least 2s of audio, or 1.0s if we want to pad early
        current_end_sec = 2.0
        window_index = 0
        
        while current_end_sec * sr <= total_samples:
            end_idx = int(current_end_sec * sr)
            
            # Other chunk: exactly 2.0s before current_end_sec (for Prosody/SV and UI display)
            other_start_sec = max(0.0, current_end_sec - other_window_sec)
            other_start_idx = int(other_start_sec * sr)
            other_chunk = audio_data[other_start_idx:end_idx]
            other_processed = preprocess_for_detection(other_chunk, sr, debug=False)
            
            # AASIST chunk: grab 4.0375s starting from the SAME start time as the UI window (Lookahead)
            # This ensures we always provide 64,600 samples of contiguous audio for AASIST
            aasist_start_idx = other_start_idx
            aasist_end_idx = min(total_samples, aasist_start_idx + int(aasist_window_sec * sr))
            aasist_chunk = audio_data[aasist_start_idx:aasist_end_idx]
            aasist_processed_raw = preprocess_for_detection(aasist_chunk, sr, debug=False)
            
            padding_mode = re.config.get("aasist", {}).get("padding", "tile")
            
            from app.preprocessing.pipeline import prepare_aasist_context
            aasist_processed = prepare_aasist_context(aasist_processed_raw, padding_mode=padding_mode)
            
            # Save the exact model input for the first window for debugging
            if window_index == 0:
                import scipy.io.wavfile as wav
                os.makedirs("debug", exist_ok=True)
                wav.write("debug/testbench_aasist_window.wav", 16000, aasist_processed)
            
            # Detect
            aasist_res = aasist.predict_aasist(aasist_processed, 16000)
            a_score = aasist_res["spoof_score"]
            p_res = prosody.score(other_processed, 16000)
            p_score = p_res["spoof_score"]
            s_score = sv.score(other_processed, enrolled_identity_id) if enrolled_identity_id else None
            
            # Fuse
            ctx_flags = {"unknown_caller": False, "high_value_keywords": False, "ivr_allowlisted": False}
            win_res = re.score_window(call_id, a_score, p_score, s_score, ctx_flags)
            
            # Send to Pipeline for WS broadcast and Audit logging
            from app.pipeline import CallPipeline
            pipeline = CallPipeline(re)
            await pipeline.process_window(call_id, win_res, "tier1", {"aasist": "v1", "prosody": "v1"})
            
            windows_result.append({
                "window_index": window_index,
                "start_s": other_start_sec,
                "end_s": current_end_sec,
                "aasist_score": a_score,
                "prosody_score": p_score,
                "speaker_score": s_score,
                "fused_score": win_res.r_final,
                "classification": win_res.classification
            })
            
            current_end_sec += step_duration_sec
            window_index += 1
            
        final_state = re.call_states.get(call_id)
        if final_state and final_state.history:
            overall_score = final_state.history[-1]
            overall_class = final_state.classification
        else:
            overall_score = 0.0
            overall_class = "LOW" 
        
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
