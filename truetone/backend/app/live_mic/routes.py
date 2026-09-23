import asyncio
import numpy as np
import time
import os
from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from app.preprocessing.pipeline import preprocess_for_detection
from app.risk_engine.fusion import RiskEngine
from app.test_bench.routes import get_models

router = APIRouter(prefix="/ws", tags=["live_mic"])

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
            
            # Fire when we have at least OTHER_SAMPLES (2.0s)
            while len(audio_buffer) >= OTHER_SAMPLES:
                # Grab up to AASIST_SAMPLES (4.04s) from history
                aasist_chunk = audio_buffer[-AASIST_SAMPLES:] if len(audio_buffer) > AASIST_SAMPLES else audio_buffer
                # Grab up to OTHER_SAMPLES (2.0s) from history
                other_chunk = audio_buffer[-OTHER_SAMPLES:]
                
                max_val = float(np.abs(other_chunk).max()) if len(other_chunk) > 0 else 0.0
                if max_val < 0.01:
                    a_score = 0.0
                    p_score = 0.0
                    s_score = None
                else:
                    aasist_processed_raw = preprocess_for_detection(aasist_chunk, SR, debug=False)
                    from app.preprocessing.pipeline import prepare_aasist_context
                    aasist_processed = prepare_aasist_context(aasist_processed_raw)
                    other_processed = preprocess_for_detection(other_chunk, SR, debug=False)
                    
                    aasist_res = aasist.predict_aasist(aasist_processed, 16000)
                    a_score = aasist_res["spoof_score"]
                    p_res = prosody.score(other_processed, 16000)
                    p_score = p_res["spoof_score"]
                    s_score = sv.score(other_processed, enrolled_identity_id) if enrolled_identity_id else None
                
                ctx_flags = {"unknown_caller": False, "high_value_keywords": False, "ivr_allowlisted": False}
                win_res = re.score_window(call_id, a_score, p_score, s_score, ctx_flags)
                call_state = re.call_states[call_id]
                
                await websocket.send_json({
                    "window_index": window_index,
                    "timestamp": time.time(),
                    "aasist_score": a_score,
                    "prosody_score": p_score,
                    "speaker_score": s_score,
                    "fused_score": win_res.r_final,
                    "classification": call_state.classification
                })
                
                window_index += 1
                
                # We consume STEP_SAMPLES from the start of the buffer
                # But wait, we want to maintain history for the NEXT window!
                # If we chop STEP_SAMPLES, we keep the overlapping part for the next loop
                audio_buffer = audio_buffer[STEP_SAMPLES:]
                
    except WebSocketDisconnect:
        pass
    except Exception as e:
        print(f"Live mic WS error: {e}")
    finally:
        if call_id in re.call_states:
            del re.call_states[call_id]
