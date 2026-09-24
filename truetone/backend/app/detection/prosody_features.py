import numpy as np
import parselmouth
from parselmouth.praat import call
import librosa

def extract_prosody_features(window: np.ndarray, sr: int = 16000) -> np.ndarray:
    """
    Extracts prosody and behavioral features from a given audio window.
    Features:
    1-3. Pitch (F0) mean, std, range
    4-5. Jitter, Shimmer
    6. Speaking rate proxy (voiced-frame ratio)
    7-8. Pause count, pause duration stats
    9. Spectral flatness (mean)
    10. Harmonic-to-noise ratio (HNR) mean
    
    Returns a 1D numpy array of length 10.
    """
    # 1. Ensure window is float64 for Parselmouth and pyworld
    y = window.astype(np.float64)
    if y.ndim > 1:
        y = y.squeeze()
        
    features = []
    
    try:
        # Create Parselmouth sound object
        snd = parselmouth.Sound(y, sampling_frequency=sr)
        
        # --- Pitch (F0) ---
        pitch = snd.to_pitch()
        pitch_values = pitch.selected_array['frequency']
        # filter out unvoiced (0)
        voiced_pitch = pitch_values[pitch_values > 0]
        
        if len(voiced_pitch) > 0:
            f0_mean = np.mean(voiced_pitch)
            f0_std = np.std(voiced_pitch)
            f0_range = np.max(voiced_pitch) - np.min(voiced_pitch)
        else:
            f0_mean, f0_std, f0_range = 0.0, 0.0, 0.0
            
        features.extend([f0_mean, f0_std, f0_range])
        
        # --- Jitter and Shimmer ---
        pointProcess = call(snd, "To PointProcess (periodic, cc)", 75, 500)
        
        try:
            jitter = call(pointProcess, "Get jitter (local)", 0, 0, 0.0001, 0.02, 1.3)
            shimmer = call([snd, pointProcess], "Get shimmer (local)", 0, 0, 0.0001, 0.02, 1.3, 1.6)
            jitter = 0.0 if np.isnan(jitter) else jitter
            shimmer = 0.0 if np.isnan(shimmer) else shimmer
        except Exception:
            jitter, shimmer = 0.0, 0.0
            
        features.extend([jitter, shimmer])
        
        # --- HNR ---
        try:
            harmonicity = call(snd, "To Harmonicity (cc)", 0.01, 75, 0.1, 1.0)
            hnr = call(harmonicity, "Get mean", 0, 0)
            hnr = 0.0 if np.isnan(hnr) else hnr
        except Exception:
            hnr = 0.0
            
        # --- VAD for Speaking rate proxy and pauses ---
        # We can use parselmouth's pitch array as a VAD proxy
        voiced_frames = pitch_values > 0
        num_voiced = np.sum(voiced_frames)
        print(f"[PROSODY DEBUG] Num voiced frames: {num_voiced} / {len(voiced_frames)}")
        voiced_ratio = num_voiced / max(1, len(voiced_frames))
        features.append(voiced_ratio) # speaking rate proxy
        
        # Pause count and duration
        # A pause is a contiguous block of unvoiced frames
        unvoiced_frames = ~voiced_frames
        # Find contiguous regions
        changes = np.diff(unvoiced_frames.astype(int))
        pause_starts = np.where(changes == 1)[0]
        pause_ends = np.where(changes == -1)[0]
        
        # Handle edges
        if unvoiced_frames[0]:
            pause_starts = np.insert(pause_starts, 0, 0)
        if unvoiced_frames[-1]:
            pause_ends = np.append(pause_ends, len(unvoiced_frames) - 1)
            
        pause_durations = pause_ends - pause_starts
        
        # Assuming frame length in parselmouth to_pitch() default is 10ms (0.01s)
        frame_period = 0.01 
        pause_durations_sec = pause_durations * frame_period
        
        # Filter micro-pauses (e.g. < 50ms)
        valid_pauses = pause_durations_sec[pause_durations_sec > 0.05]
        pause_count = len(valid_pauses)
        pause_mean_dur = np.mean(valid_pauses) if pause_count > 0 else 0.0
        
        features.extend([float(pause_count), pause_mean_dur])
        
        # --- Spectral flatness ---
        S, phase = librosa.magphase(librosa.stft(y))
        flatness = librosa.feature.spectral_flatness(S=S)
        flatness_mean = np.mean(flatness)
        features.append(flatness_mean)
        
        features.append(hnr) # HNR from earlier, position 10
        
        # features length should be 10: 
        # f0_mean, f0_std, f0_range, jitter, shimmer, voiced_ratio, pause_count, pause_mean_dur, flatness, hnr
        
        return np.array(features, dtype=np.float32)

    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"Error in extract_prosody_features: {e}")
        raise e

