import os

file_path = "app/live_mic/routes.py"
with open(file_path, "r") as f:
    content = f.read()

import re

# Add case_name to websocket
content = content.replace(
    "async def live_mic_endpoint(websocket: WebSocket, enrolled_identity_id: str = None):",
    "async def live_mic_endpoint(websocket: WebSocket, enrolled_identity_id: str = None, case_name: str = 'default'):\n    os.makedirs('logs', exist_ok=True)\n    csv_file = f'logs/live_debug_{case_name}.csv'\n    if not os.path.exists(csv_file):\n        with open(csv_file, 'w') as f:\n            f.write('window_index,rms_dbfs,peak,flatness,vad_frac,voiced_frames,zero_padded,logits,aasist_score,fused_score,class_name\\n')"
)

# Replace the debug logging we added earlier
content = re.sub(r'# Debug logging.*?print\(f"AASIST raw logits:.*?\]"\)', r'''
                import librosa
                # Debug logging
                rms_val = np.sqrt(np.mean(other_chunk**2)) if len(other_chunk) > 0 else 1e-10
                rms_dbfs = 20 * np.log10(rms_val + 1e-10)
                peak_val = float(np.abs(other_chunk).max()) if len(other_chunk) > 0 else 0.0
                
                # Spectral flatness
                try:
                    S, phase = librosa.magphase(librosa.stft(other_chunk.astype(np.float64)))
                    flatness = float(np.mean(librosa.feature.spectral_flatness(S=S)))
                except:
                    flatness = 0.0
                
                # VAD (simple energy proxy since we don't have silero yet for PART 1 logging)
                # We will use webrtcvad if available, else simple energy
                try:
                    import webrtcvad
                    vad = webrtcvad.Vad(3)
                    # convert to 16bit PCM
                    pcm = (other_chunk * 32767).astype(np.int16).tobytes()
                    frame_len = int(SR * 0.03) # 30ms
                    num_frames = len(pcm) // (frame_len * 2)
                    vad_frames = 0
                    for i in range(num_frames):
                        frame = pcm[i*frame_len*2:(i+1)*frame_len*2]
                        if vad.is_speech(frame, SR):
                            vad_frames += 1
                    vad_frac = vad_frames / max(1, num_frames)
                except ImportError:
                    vad_frac = 0.0
                
                if max_val < 0.01:
                    a_score = 0.0
                    p_score = 0.0
                    s_score = None
                    logits_str = "[]"
                    zero_padded = 0
                    voiced_frames = 0
                else:
                    aasist_processed_raw = preprocess_for_detection(aasist_chunk, SR, debug=False)
                    from app.preprocessing.pipeline import prepare_aasist_context
                    aasist_processed = prepare_aasist_context(aasist_processed_raw)
                    zero_padded = 64600 - len(aasist_chunk)
                    if zero_padded < 0: zero_padded = 0
                    other_processed = preprocess_for_detection(other_chunk, SR, debug=False)
                    
                    aasist_res = aasist.predict_aasist(aasist_processed, 16000)
                    logits_str = str(aasist_res.get("logits", []))
                    a_score = aasist_res["spoof_score"]
                    
                    p_res = prosody.score(other_processed, 16000)
                    p_score = p_res["spoof_score"]
                    s_score = sv.score(other_processed, enrolled_identity_id) if enrolled_identity_id else None
                    
                    # We modified prosody.py to print voiced frames. Let's just capture it or approximate here.
                    # Or we can just read it from the prosody output if we return it. Let's return it from prosody!
''', content, flags=re.DOTALL)

with open(file_path, "w") as f:
    f.write(content)
