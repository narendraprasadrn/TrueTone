import numpy as np
import parselmouth
from parselmouth.praat import call
import librosa
import traceback

def extract_features_v3(window: np.ndarray, sr: int = 16000) -> dict:
    y = window.astype(np.float64)
    if y.ndim > 1:
        y = y.squeeze()
        
    features = {}
    
    # Short-audio fallback: If audio is too short for librosa's STFT with n_fft=2048,
    # pad it with zeros at the end to avoid librosa throwing exceptions.
    # We do NOT pad for Parselmouth (prosody) as it handles short audio fine internally 
    # and zero padding could distort pitch statistics.
    y_padded = y
    if len(y_padded) < 2048:
        y_padded = np.pad(y_padded, (0, 2048 - len(y_padded)), mode='constant')
        
    # 1. Prosodic & Voice Quality & Behavioural (Parselmouth & heuristics)
    try:
        snd = parselmouth.Sound(y, sampling_frequency=sr)
        pitch = snd.to_pitch()
        pitch_values = pitch.selected_array['frequency']
        voiced_pitch = pitch_values[pitch_values > 0]
        
        if len(voiced_pitch) > 0:
            features['f0_mean'] = np.mean(voiced_pitch)
            features['f0_std'] = np.std(voiced_pitch)
            features['f0_range'] = np.max(voiced_pitch) - np.min(voiced_pitch)
            features['f0_median'] = np.median(voiced_pitch)
            features['f0_min'] = np.min(voiced_pitch)
            features['f0_max'] = np.max(voiced_pitch)
        else:
            for k in ['f0_mean', 'f0_std', 'f0_range', 'f0_median', 'f0_min', 'f0_max']:
                features[k] = 0.0
            
        pointProcess = call(snd, "To PointProcess (periodic, cc)", 75, 500)
        try:
            jitter = call(pointProcess, "Get jitter (local)", 0, 0, 0.0001, 0.02, 1.3)
            shimmer = call([snd, pointProcess], "Get shimmer (local)", 0, 0, 0.0001, 0.02, 1.3, 1.6)
            features['jitter'] = 0.0 if np.isnan(jitter) else jitter
            features['shimmer'] = 0.0 if np.isnan(shimmer) else shimmer
        except Exception:
            features['jitter'], features['shimmer'] = 0.0, 0.0
            
        try:
            harmonicity = call(snd, "To Harmonicity (cc)", 0.01, 75, 0.1, 1.0)
            hnr = call(harmonicity, "Get mean", 0, 0)
            features['hnr'] = 0.0 if np.isnan(hnr) else hnr
        except Exception:
            features['hnr'] = 0.0
            
        voiced_frames = pitch_values > 0
        features['voiced_ratio'] = np.sum(voiced_frames) / max(1, len(voiced_frames))
        
        # Pauses calculation
        unvoiced_frames = ~voiced_frames
        changes_unv = np.diff(unvoiced_frames.astype(int))
        pause_starts = np.where(changes_unv == 1)[0]
        pause_ends = np.where(changes_unv == -1)[0]
        
        if len(unvoiced_frames) > 0:
            if unvoiced_frames[0]: pause_starts = np.insert(pause_starts, 0, 0)
            if unvoiced_frames[-1]: pause_ends = np.append(pause_ends, len(unvoiced_frames) - 1)
        
        pause_durations_sec = (pause_ends - pause_starts) * 0.01 
        valid_pauses = pause_durations_sec[pause_durations_sec > 0.05]
        
        features['pause_count'] = float(len(valid_pauses))
        features['pause_mean_dur'] = np.mean(valid_pauses) if len(valid_pauses) > 0 else 0.0
        features['pause_max_dur'] = np.max(valid_pauses) if len(valid_pauses) > 0 else 0.0
        features['pause_median_dur'] = np.median(valid_pauses) if len(valid_pauses) > 0 else 0.0
        
        features['speech_duration'] = np.sum(voiced_frames) * 0.01
        features['silence_duration'] = np.sum(unvoiced_frames) * 0.01
        features['speech_silence_ratio'] = features['speech_duration'] / max(0.01, features['silence_duration'])
        
        # Speech segments calculation (NEW in V3)
        changes_v = np.diff(voiced_frames.astype(int))
        speech_starts = np.where(changes_v == 1)[0]
        speech_ends = np.where(changes_v == -1)[0]
        
        if len(voiced_frames) > 0:
            if voiced_frames[0]: speech_starts = np.insert(speech_starts, 0, 0)
            if voiced_frames[-1]: speech_ends = np.append(speech_ends, len(voiced_frames) - 1)
            
        speech_durations_sec = (speech_ends - speech_starts) * 0.01
        valid_speech = speech_durations_sec[speech_durations_sec > 0.0]
        
        features['speech_segment_count'] = float(len(valid_speech))
        features['temporal_regularity'] = np.var(valid_speech) if len(valid_speech) > 0 else 0.0
        
    except Exception as e:
        # Fallback values for prosody and behavioural
        for k in ['f0_mean','f0_std','f0_range','f0_median','f0_min','f0_max','jitter','shimmer','hnr','voiced_ratio',
                 'pause_count','pause_mean_dur','pause_max_dur','pause_median_dur','speech_duration','silence_duration',
                 'speech_silence_ratio','speech_segment_count','temporal_regularity']:
            features[k] = 0.0

    # 2. Time/Energy (using padded audio to avoid exceptions)
    try:
        rms = librosa.feature.rms(y=y_padded)[0]
        features['rms_mean'] = np.mean(rms)
        features['rms_std'] = np.std(rms)
        features['rms_min'] = np.min(rms)
        features['rms_max'] = np.max(rms)

        zcr = librosa.feature.zero_crossing_rate(y_padded)[0]
        features['zcr_mean'] = np.mean(zcr)
        features['zcr_std'] = np.std(zcr)

        # 3. Spectral Ext
        cent = librosa.feature.spectral_centroid(y=y_padded, sr=sr)[0]
        features['spectral_centroid_mean'] = np.mean(cent)
        features['spectral_centroid_std'] = np.std(cent)

        bw = librosa.feature.spectral_bandwidth(y=y_padded, sr=sr)[0]
        features['spectral_bandwidth_mean'] = np.mean(bw)
        features['spectral_bandwidth_std'] = np.std(bw)

        rolloff = librosa.feature.spectral_rolloff(y=y_padded, sr=sr)[0]
        features['spectral_rolloff_mean'] = np.mean(rolloff)
        features['spectral_rolloff_std'] = np.std(rolloff)
        
        S, phase = librosa.magphase(librosa.stft(y_padded))
        flatness = librosa.feature.spectral_flatness(S=S)[0]
        features['flatness_mean'] = np.mean(flatness)
        features['flatness_std'] = np.std(flatness)
        
        flux = np.mean(np.diff(S, axis=1)**2, axis=0)
        features['spectral_flux_mean'] = np.mean(flux) if len(flux) > 0 else 0.0

        # 4. MFCC (13) + Delta + Delta-Delta
        mfcc = librosa.feature.mfcc(y=y_padded, sr=sr, n_mfcc=13)
        n_frames = mfcc.shape[1]
        width = min(9, n_frames if n_frames % 2 != 0 else n_frames - 1)
        
        if width < 3:
            mfcc_delta = np.zeros_like(mfcc)
            mfcc_delta2 = np.zeros_like(mfcc)
        else:
            mfcc_delta = librosa.feature.delta(mfcc, width=width)
            mfcc_delta2 = librosa.feature.delta(mfcc, order=2, width=width)

        for i in range(13):
            features[f'mfcc_{i}_mean'] = np.mean(mfcc[i])
            features[f'mfcc_{i}_std'] = np.std(mfcc[i])
            features[f'mfcc_delta_{i}_mean'] = np.mean(mfcc_delta[i])
            features[f'mfcc_delta_{i}_std'] = np.std(mfcc_delta[i])
            features[f'mfcc_delta2_{i}_mean'] = np.mean(mfcc_delta2[i])
            features[f'mfcc_delta2_{i}_std'] = np.std(mfcc_delta2[i])
            
    except Exception as e:
        for k in ['rms_mean','rms_std','rms_min','rms_max','zcr_mean','zcr_std',
                 'spectral_centroid_mean','spectral_centroid_std','spectral_bandwidth_mean','spectral_bandwidth_std',
                 'spectral_rolloff_mean','spectral_rolloff_std','flatness_mean','flatness_std','spectral_flux_mean']:
            features[k] = 0.0
        for i in range(13):
            features[f'mfcc_{i}_mean'] = 0.0
            features[f'mfcc_{i}_std'] = 0.0
            features[f'mfcc_delta_{i}_mean'] = 0.0
            features[f'mfcc_delta_{i}_std'] = 0.0
            features[f'mfcc_delta2_{i}_mean'] = 0.0
            features[f'mfcc_delta2_{i}_std'] = 0.0
            
    # Guarantee deterministic ordering based on v3 feature names
    import json
    import os
    names_path = os.path.join(os.path.dirname(__file__), 'feature_names_v3.json')
    with open(names_path, 'r') as f:
        v3_names = [record['name'] for record in json.load(f)]
        
    ordered_features = {name: features.get(name, 0.0) for name in v3_names}
    return ordered_features

