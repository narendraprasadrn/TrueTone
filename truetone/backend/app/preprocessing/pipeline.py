import numpy as np
import librosa

def preprocess_for_detection(raw_audio: np.ndarray, source_sr: int, debug: bool = False) -> np.ndarray:
    """
    Unifies audio preprocessing.
    - convert to mono
    - resample to 16000
    - cast to float32
    - reject NaN/Inf
    - remove DC offset
    - NO aggressive denoise, NO arbitrary min/max normalization
    """
    audio = raw_audio
    
    # 1. Reject NaN/Inf
    if not np.isfinite(audio).all():
        audio = np.nan_to_num(audio)
        
    # 2. Mono
    if audio.ndim > 1:
        audio = librosa.to_mono(audio.T if audio.shape[1] > audio.shape[0] else audio)
        
    # 3. Resample
    if source_sr != 16000:
        audio = librosa.resample(y=audio, orig_sr=source_sr, target_sr=16000)
        
    # 4. Convert to float32
    audio = audio.astype(np.float32)
    
    # 5. Remove DC offset
    audio = audio - np.mean(audio)
    
    if debug:
        print(f"[PREPROCESS DEBUG] output_samples={len(audio)}, sr=16000, rms={np.sqrt(np.mean(audio**2)):.4f}")
        
    return audio

def prepare_aasist_context(audio_buffer: np.ndarray) -> np.ndarray:
    """
    Prepares a contiguous 64600-sample context for AASIST-L.
    If the buffer is shorter than 64600 samples, it is zero-padded at the beginning (pre-padding).
    If it is longer, the most recent 64600 samples are taken.
    NEVER uses periodic repetition (np.tile).
    """
    REQUIRED_SAMPLES = 64600
    
    if len(audio_buffer) >= REQUIRED_SAMPLES:
        context = audio_buffer[-REQUIRED_SAMPLES:]
    else:
        # Zero pad at the beginning
        pad_width = REQUIRED_SAMPLES - len(audio_buffer)
        context = np.pad(audio_buffer, (pad_width, 0), mode='constant', constant_values=0.0)
        
    assert len(context) == REQUIRED_SAMPLES, f"Context length must be exactly {REQUIRED_SAMPLES}, got {len(context)}"
    return context
