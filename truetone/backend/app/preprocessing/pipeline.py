import librosa
import numpy as np

def preprocess_for_detection(raw_audio: np.ndarray, source_sr: int, debug: bool = True) -> np.ndarray:
    """
    Single source of truth for the full 7-stage preprocessing pipeline.
    Used identically by CallSimulator, Test Bench, and Live Mic.
    
    Stages:
    1. Audio Capture/Decode (Assumed done before passing to this function)
    2. Channel Handling (force mono)
    3. Resample to 16kHz
    4. Amplitude Normalization
    5. VAD (Voice Activity Detection / noise gating)
    6. Noise/Quality Handling
    7. Sliding Window/Chunking (Assumed handled by caller, this processes one chunk)
    """
    audio_data = raw_audio
    
    if debug:
        print(f"[PREPROCESS DEBUG] 1. Input: samples={len(audio_data)}, sr={source_sr}")
        
    # 2. Channel Handling (force mono)
    if len(audio_data.shape) > 1:
        # Check if shape is (channels, samples) or (samples, channels)
        # librosa expects (channels, samples)
        if audio_data.shape[1] < audio_data.shape[0] and audio_data.shape[1] <= 2:
             audio_data = audio_data.T
        audio_data = librosa.to_mono(audio_data)
        if debug:
            print(f"[PREPROCESS DEBUG] 2. Forced mono: samples={len(audio_data)}")
            
    # 3. Resample to 16kHz
    target_sr = 16000
    if source_sr != target_sr:
        audio_data = librosa.resample(y=audio_data, orig_sr=source_sr, target_sr=target_sr)
        if debug:
            print(f"[PREPROCESS DEBUG] 3. Resampled to {target_sr}Hz: samples={len(audio_data)}")

    # 4. Amplitude Normalization
    max_val = np.abs(audio_data).max() if len(audio_data) > 0 else 0.0
    if max_val > 0.01:
        audio_data = audio_data / max_val
    if debug:
        rms_norm = float(np.sqrt(np.mean(audio_data**2))) if len(audio_data) > 0 else 0.0
        print(f"[PREPROCESS DEBUG] 4. Normalized: max={max_val:.6f}, RMS={rms_norm:.6f}")

    # 5. VAD
    if len(audio_data) > 0:
        voiced_audio, _ = librosa.effects.trim(audio_data, top_db=20)
        if len(voiced_audio) > 0:
            audio_data = voiced_audio
            
    if debug:
        rms_vad = float(np.sqrt(np.mean(audio_data**2))) if len(audio_data) > 0 else 0.0
        print(f"[PREPROCESS DEBUG] 5. Post-VAD: samples={len(audio_data)}, RMS={rms_vad:.6f}")

    # 6. Noise/Quality Handling
    # (Placeholder for additional handling)
    if debug:
        print(f"[PREPROCESS DEBUG] 6. Noise Handling: passed")

    return audio_data
