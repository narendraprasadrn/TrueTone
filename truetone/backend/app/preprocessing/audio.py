import librosa
import numpy as np

def preprocess_audio(audio_data: np.ndarray, sr: int, target_sr: int = 16000, vad_top_db: int = 20) -> np.ndarray:
    """
    Resample to 16kHz mono, normalize, and trim silence using librosa.
    audio_data: numpy array of audio samples (float32, -1.0 to 1.0)
    sr: original sample rate
    vad_top_db: decibel threshold for silence trimming. If None, skips VAD entirely.
    """
    # 1. Resample to target_sr if necessary
    if sr != target_sr:
        audio_data = librosa.resample(y=audio_data, orig_sr=sr, target_sr=target_sr)
        
    rms_post_resample = float(np.sqrt(np.mean(audio_data**2))) if len(audio_data) > 0 else 0.0
    print(f"[DEBUG PREPROCESS] Post-resample: samples={len(audio_data)}, RMS={rms_post_resample:.6f}")
    
    # 2. Normalize audio to -1.0 to 1.0, but ONLY if it's loud enough
    max_val = np.abs(audio_data).max()
    if max_val > 0.01:
        audio_data = audio_data / max_val
        
    rms_post_norm = float(np.sqrt(np.mean(audio_data**2))) if len(audio_data) > 0 else 0.0
    print(f"[DEBUG PREPROCESS] Post-normalize: samples={len(audio_data)}, RMS={rms_post_norm:.6f}, max_val_used={max_val:.6f}")

    if vad_top_db is None:
        print("[DEBUG PREPROCESS] Skipping VAD trim (live mode).")
        return audio_data

    # 3. VAD (trim silence) using librosa
    voiced_audio, _ = librosa.effects.trim(audio_data, top_db=vad_top_db)
    
    rms_post_vad = float(np.sqrt(np.mean(voiced_audio**2))) if len(voiced_audio) > 0 else 0.0
    print(f"[DEBUG PREPROCESS] Post-VAD trim: samples={len(voiced_audio)} (was {len(audio_data)}), RMS={rms_post_vad:.6f}")
    
    if len(voiced_audio) == 0:
        print("[DEBUG PREPROCESS] VAD trimmed everything! Returning original normalized audio.")
        return audio_data
        
    return voiced_audio
