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
    WINDOW_SEC = 2.0
    STEP_SEC = 1.0
    WINDOW_SAMPLES = int(WINDOW_SEC * SR)
    STEP_SAMPLES = int(STEP_SEC * SR)
    
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
                print(f"[LIVE-MIC DEBUG] received message: type={type(data)}, size={len(data) if data else 0} bytes")
            except asyncio.TimeoutError:
                continue
                
            chunk = np.frombuffer(data, dtype=np.float32)
            audio_buffer = np.concatenate((audio_buffer, chunk))
            
            while len(audio_buffer) >= WINDOW_SAMPLES:
                window_chunk = audio_buffer[:WINDOW_SAMPLES]
                
                rms = float(np.sqrt(np.mean(window_chunk**2))) if len(window_chunk) > 0 else 0.0
                
                # Diagnostics for Step 1
                try:
                    print(f"[DEBUG LIVE-MIC] Window {window_index}: chunk_size={len(data)} bytes, "
                          f"samples={len(window_chunk)}, duration={len(window_chunk)/SR:.2f}s, dtype={window_chunk.dtype}, "
                          f"min={float(np.min(window_chunk)):.6f}, max={float(np.max(window_chunk)):.6f}, "
                          f"mean={float(np.mean(window_chunk)):.6f}, RMS={rms:.6f}")
                except Exception as e:
                    pass

                # If the window is basically silence/room noise (e.g. peak < 0.01 = -40dBFS), bypass models
                max_val = float(np.abs(window_chunk).max()) if len(window_chunk) > 0 else 0.0
                if max_val < 0.01:
                    print(f"[DEBUG LIVE-MIC] Window {window_index}: Signal too quiet (max_val={max_val:.4f}). Returning 0.0")
                    a_score = 0.0
                    p_score = 0.0
                    s_score = None
                else:
                    rms_raw = float(np.sqrt(np.mean(window_chunk**2))) if len(window_chunk) > 0 else 0.0
                    print(f"[LIVE-MIC DEBUG] raw buffer RMS: {rms_raw:.6f}")
                    
                    processed = preprocess_for_detection(window_chunk, SR, debug=True)
                    
                    rms_trimmed = float(np.sqrt(np.mean(processed**2))) if len(processed) > 0 else 0.0
                    print(f"[LIVE-MIC DEBUG] post-VAD sample count: {len(processed)} (was {len(window_chunk)} before VAD)")
                    print(f"[LIVE-MIC DEBUG] post-VAD RMS: {rms_trimmed:.6f}")
                    
                    if len(processed) == 0:
                        print(f"[WARNING] Window {window_index}: processed audio is EMPTY!")
                    
                    a_score = aasist.score(processed, 16000)
                    p_score = prosody.score(processed, 16000)
                    s_score = sv.score(processed, enrolled_identity_id) if enrolled_identity_id else None
                
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
                audio_buffer = audio_buffer[STEP_SAMPLES:]
                
    except WebSocketDisconnect:
        pass
    except Exception as e:
        print(f"Live mic WS error: {e}")
    finally:
        if call_id in re.call_states:
            del re.call_states[call_id]
