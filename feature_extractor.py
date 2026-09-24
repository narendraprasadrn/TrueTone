import numpy as np
import parselmouth
from parselmouth.praat import call
import librosa
import traceback

def extract_features_v2(window: np.ndarray, sr: int = 16000) -> dict:
    y = window.astype(np.float64)
    if y.ndim > 1:
        y = y.squeeze()
        
    features = {}
    
    # 1. Existing 10 features for backwards compatibility / baseline test
    try:
        snd = parselmouth.Sound(y, sampling_frequency=sr)
        pitch = snd.to_pitch()
        pitch_values = pitch.selected_array['frequency']
        voiced_pitch = pitch_values[pitch_values > 0]
        
        if len(voiced_pitch) > 0:
            f0_mean = np.mean(voiced_pitch)
            f0_std = np.std(voiced_pitch)
            f0_range = np.max(voiced_pitch) - np.min(voiced_pitch)
            f0_median = np.median(voiced_pitch)
            f0_min = np.min(voiced_pitch)
            f0_max = np.max(voiced_pitch)
        else:
            f0_mean, f0_std, f0_range, f0_median, f0_min, f0_max = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
            
        features['f0_mean'] = f0_mean
        features['f0_std'] = f0_std
        features['f0_range'] = f0_range
        features['f0_median'] = f0_median
        features['f0_min'] = f0_min
        features['f0_max'] = f0_max
        
        pointProcess = call(snd, "To PointProcess (periodic, cc)", 75, 500)
        try:
            jitter = call(pointProcess, "Get jitter (local)", 0, 0, 0.0001, 0.02, 1.3)
            shimmer = call([snd, pointProcess], "Get shimmer (local)", 0, 0, 0.0001, 0.02, 1.3, 1.6)
            jitter = 0.0 if np.isnan(jitter) else jitter
            shimmer = 0.0 if np.isnan(shimmer) else shimmer
        except Exception:
            jitter, shimmer = 0.0, 0.0
            
        features['jitter'] = jitter
        features['shimmer'] = shimmer
        
        try:
            harmonicity = call(snd, "To Harmonicity (cc)", 0.01, 75, 0.1, 1.0)
            hnr = call(harmonicity, "Get mean", 0, 0)
            hnr = 0.0 if np.isnan(hnr) else hnr
        except Exception:
            hnr = 0.0
        features['hnr'] = hnr
        
        voiced_frames = pitch_values > 0
        voiced_ratio = np.sum(voiced_frames) / max(1, len(voiced_frames))
        features['voiced_ratio'] = voiced_ratio
        
        unvoiced_frames = ~voiced_frames
        changes = np.diff(unvoiced_frames.astype(int))
        pause_starts = np.where(changes == 1)[0]
        pause_ends = np.where(changes == -1)[0]
        
        if len(unvoiced_frames) > 0:
            if unvoiced_frames[0]: pause_starts = np.insert(pause_starts, 0, 0)
            if unvoiced_frames[-1]: pause_ends = np.append(pause_ends, len(unvoiced_frames) - 1)
        
        pause_durations_sec = (pause_ends - pause_starts) * 0.01 
        valid_pauses = pause_durations_sec[pause_durations_sec > 0.05]
        
        features['pause_count'] = float(len(valid_pauses))
        features['pause_mean_dur'] = np.mean(valid_pauses) if len(valid_pauses) > 0 else 0.0
        features['pause_max_dur'] = np.max(valid_pauses) if len(valid_pauses) > 0 else 0.0
        features['speech_duration'] = np.sum(voiced_frames) * 0.01
        features['silence_duration'] = np.sum(unvoiced_frames) * 0.01
        features['speech_silence_ratio'] = features['speech_duration'] / max(0.01, features['silence_duration'])
        
        # Spectrals
        S, phase = librosa.magphase(librosa.stft(y))
        flatness = librosa.feature.spectral_flatness(S=S)[0]
        features['flatness_mean'] = np.mean(flatness)
        features['flatness_std'] = np.std(flatness)
        
    except Exception as e:
        # Fallback values
        for k in ['f0_mean','f0_std','f0_range','f0_median','f0_min','f0_max','jitter','shimmer','hnr','voiced_ratio',
                 'pause_count','pause_mean_dur','pause_max_dur','speech_duration','silence_duration','speech_silence_ratio',
                 'flatness_mean','flatness_std']:
            features[k] = 0.0

    # 2. Time/Energy
    rms = librosa.feature.rms(y=y)[0]
    features['rms_mean'] = np.mean(rms)
    features['rms_std'] = np.std(rms)
    features['rms_min'] = np.min(rms)
    features['rms_max'] = np.max(rms)

    zcr = librosa.feature.zero_crossing_rate(y)[0]
    features['zcr_mean'] = np.mean(zcr)
    features['zcr_std'] = np.std(zcr)

    # 3. Spectral Ext
    cent = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
    features['spectral_centroid_mean'] = np.mean(cent)
    features['spectral_centroid_std'] = np.std(cent)

    bw = librosa.feature.spectral_bandwidth(y=y, sr=sr)[0]
    features['spectral_bandwidth_mean'] = np.mean(bw)
    features['spectral_bandwidth_std'] = np.std(bw)

    rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)[0]
    features['spectral_rolloff_mean'] = np.mean(rolloff)
    features['spectral_rolloff_std'] = np.std(rolloff)
    
    # Spectral flux
    if 'S' not in locals():
        S, _ = librosa.magphase(librosa.stft(y))
    flux = np.mean(np.diff(S, axis=1)**2, axis=0)
    features['spectral_flux_mean'] = np.mean(flux) if len(flux) > 0 else 0.0

    # 4. MFCC (13) + Delta + Delta-Delta
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    mfcc_delta = librosa.feature.delta(mfcc)
    mfcc_delta2 = librosa.feature.delta(mfcc, order=2)

    for i in range(13):
        features[f'mfcc_{i}_mean'] = np.mean(mfcc[i])
        features[f'mfcc_{i}_std'] = np.std(mfcc[i])
        features[f'mfcc_delta_{i}_mean'] = np.mean(mfcc_delta[i])
        features[f'mfcc_delta_{i}_std'] = np.std(mfcc_delta[i])
        features[f'mfcc_delta2_{i}_mean'] = np.mean(mfcc_delta2[i])
        features[f'mfcc_delta2_{i}_std'] = np.std(mfcc_delta2[i])
        
    return features

# Helper to maintain the exact 10 old features in exact order
def extract_old_10_features(features_dict: dict) -> list:
    return [
        features_dict.get('f0_mean', 0.0),
        features_dict.get('f0_std', 0.0),
        features_dict.get('f0_range', 0.0),
        features_dict.get('jitter', 0.0),
        features_dict.get('shimmer', 0.0),
        features_dict.get('voiced_ratio', 0.0),
        features_dict.get('pause_count', 0.0),
        features_dict.get('pause_mean_dur', 0.0),
        features_dict.get('flatness_mean', 0.0),
        features_dict.get('hnr', 0.0)
    ]
