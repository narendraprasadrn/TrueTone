import asyncio
import numpy as np
import time
import os
import torch
import librosa
from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from app.preprocessing.pipeline import preprocess_for_detection
from app.risk_engine.fusion import RiskEngine
from app.test_bench.routes import get_models

router = APIRouter(prefix="/ws", tags=["live_mic"])

# Load Silero VAD globally
try:
    vad_model, utils = torch.hub.load(repo_or_dir='snakers4/silero-vad', model='silero_vad', force_reload=False, onnx=False, trust_repo=True)
except TypeError:
    vad_model, utils = torch.hub.load(repo_or_dir='snakers4/silero-vad', model='silero_vad', force_reload=False, onnx=False)
(get_speech_timestamps, _, read_audio, _, _) = utils

@router.websocket("/live-mic")
async def live_mic_endpoint(websocket: WebSocket, enrolled_identity_id: str = None):
    await websocket.accept()
    
    try:
        aasist, prosody, sv, _ = get_models()
    except Exception as e:
        await websocket.close(code=1011, reason=f"Model init failed: {e}")
        return
        
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    re = RiskEngine(os.path.join(base_dir, "app/risk_engine/config.yaml"))
    
    # Load VAD config
    vad_cfg = re.config.get("vad", {})
    min_speech_seconds = vad_cfg.get("min_speech_seconds", 1.0)
    vad_threshold = vad_cfg.get("vad_threshold", 0.5)
    flatness_max = vad_cfg.get("flatness_max", 0.05)
    
    call_id = f"live_{id(websocket)}"
    
    SR = 16000
    STEP_SEC = 1.0
    AASIST_SEC = 64600 / 16000 # 4.0375
    OTHER_SEC = 2.0
    
    STEP_SAMPLES = int(STEP_SEC * SR)
    AASIST_SAMPLES = 64600
    OTHER_SAMPLES = int(OTHER_SEC * SR)
    
    audio_buffer = np.array([], dtype=np.float32)
    window_index = 0
    start_time = time.time()
    
    try:
        while True:
            if time.time() - start_time > 300:
                await websocket.send_json({"error": "Session length cap reached."})
                break
                
            try:
                data = await asyncio.wait_for(websocket.receive_bytes(), timeout=5.0)
            except asyncio.TimeoutError:
                continue
                
            chunk = np.frombuffer(data, dtype=np.float32)
            audio_buffer = np.concatenate((audio_buffer, chunk))
            
            # Fire only when we have enough for BOTH models (AASIST needs 64600 samples)
            while len(audio_buffer) >= max(AASIST_SAMPLES, OTHER_SAMPLES):
                
                aasist_chunk = audio_buffer[-AASIST_SAMPLES:]
                other_chunk = audio_buffer[-OTHER_SAMPLES:]
                
                # Spectral flatness
                try:
                    S, _ = librosa.magphase(librosa.stft(other_chunk.astype(np.float64)))
                    flatness = float(np.mean(librosa.feature.spectral_flatness(S=S)))
                except Exception:
                    flatness = 0.0
                
                # Silero VAD
                # Convert to tensor and get speech timestamps
                tensor_chunk = torch.from_numpy(other_chunk)
                speech_timestamps = get_speech_timestamps(tensor_chunk, vad_model, sampling_rate=SR, threshold=vad_threshold)
                
                speech_samples = sum([t['end'] - t['start'] for t in speech_timestamps])
                speech_seconds = speech_samples / SR
                
                if speech_seconds < min_speech_seconds or flatness > flatness_max:
                    # Gate failed
                    await websocket.send_json({
                        "window_index": window_index,
                        "timestamp": time.time(),
                        "aasist_score": 0.0,
                        "prosody_score": 0.0,
                        "speaker_score": None,
                        "fused_score": 0.0,
                        "classification": "no_speech"
                    })
                else:
                    # Passed gate
                    # No zero pre-padding since len(aasist_chunk) == 64600
                    aasist_processed_raw = preprocess_for_detection(aasist_chunk, SR, debug=False)
                    from app.preprocessing.pipeline import prepare_aasist_context
                    aasist_processed = prepare_aasist_context(aasist_processed_raw)
                    other_processed = preprocess_for_detection(other_chunk, SR, debug=False)
                    
                    aasist_res = aasist.predict_aasist(aasist_processed, 16000)
                    a_score = aasist_res["spoof_score"]
                    
                    p_res = prosody.score(other_processed, 16000)
                    if p_res is None:
                        p_score = None
                    else:
                        p_score = p_res["spoof_score"]
                        
                    s_score = sv.score(other_processed, enrolled_identity_id) if enrolled_identity_id else None
                
                    ctx_flags = {"unknown_caller": False, "high_value_keywords": False, "ivr_allowlisted": False}
                    win_res = re.score_window(call_id, a_score, p_score, s_score, ctx_flags)
                    call_state = re.call_states[call_id]
                    
                    await websocket.send_json({
                        "window_index": window_index,
                        "timestamp": time.time(),
                        "aasist_score": a_score,
                        "prosody_score": p_score if p_score is not None else -1,
                        "speaker_score": s_score,
                        "fused_score": win_res.r_final,
                        "classification": call_state.classification,
                        "reason": "insufficient_voiced_frames" if p_score is None else None
                    })
                
                window_index += 1
                audio_buffer = audio_buffer[STEP_SAMPLES:]
                
    except WebSocketDisconnect:
        pass
    except Exception as e:
        print(f"Live mic WS error: {e}")
    finally:
        if call_id in re.call_states:
            del re.call_states[call_id]
