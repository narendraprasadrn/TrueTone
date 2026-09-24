import numpy as np
import pandas as pd
import librosa
import glob
import os
import soundfile as sf
import json
from feature_extractor import extract_features_v2

# Re-implement V3 here temporarily to verify

def extract_features_v3(window: np.ndarray, sr: int = 16000) -> dict:
    y = window.astype(np.float64)
    if y.ndim > 1:
        y = y.squeeze()
        
    # Short audio fallback for STFT operations (requires n_fft=2048)
    if len(y) < 2048:
        # Pad center ensures signal doesn't abruptly start/stop at boundaries if it's very small
        y_padded = librosa.util.pad_center(data=y, size=2048)
    else:
        y_padded = y
        
    features = {}
    import parselmouth
    from parselmouth.praat import call
    
    # 1. Prosodic & Voice Quality & Behavioural
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
            features['f0_mean'], features['f0_std'], features['f0_range'] = 0.0, 0.0, 0.0
            features['f0_median'], features['f0_min'], features['f0_max'] = 0.0, 0.0, 0.0
            
        pointProcess = call(snd, "To PointProcess (periodic, cc)", 75, 500)
        try:
            jitter = call(pointProcess, "Get jitter (local)", 0, 0, 0.0001, 0.02, 1.3)
            shimmer = call([snd, pointProcess], "Get shimmer (local)", 0, 0, 0.0001, 0.02, 1.3, 1.6)
            features['jitter'] = 0.0 if np.isnan(jitter) else jitter
            features['shimmer'] = 0.0 if np.isnan(shimmer) else shimmer
        except:
            features['jitter'], features['shimmer'] = 0.0, 0.0
            
        try:
            harmonicity = call(snd, "To Harmonicity (cc)", 0.01, 75, 0.1, 1.0)
            hnr = call(harmonicity, "Get mean", 0, 0)
            features['hnr'] = 0.0 if np.isnan(hnr) else hnr
        except:
            features['hnr'] = 0.0
            
        voiced_frames = pitch_values > 0
        features['voiced_ratio'] = np.sum(voiced_frames) / max(1, len(voiced_frames))
        
        unvoiced_frames = ~voiced_frames
        
        # Calculate pauses
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
        
        # Calculate speech segments
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

        # Spectrals (using padded to avoid exceptions)
        S, phase = librosa.magphase(librosa.stft(y_padded))
        flatness = librosa.feature.spectral_flatness(S=S)[0]
        features['flatness_mean'] = np.mean(flatness)
        features['flatness_std'] = np.std(flatness)
        
    except Exception as e:
        for k in ['f0_mean','f0_std','f0_range','f0_median','f0_min','f0_max','jitter','shimmer','hnr','voiced_ratio',
                 'pause_count','pause_mean_dur','pause_max_dur','pause_median_dur','speech_duration','silence_duration',
                 'speech_silence_ratio','speech_segment_count','temporal_regularity','flatness_mean','flatness_std']:
            features[k] = 0.0

    # 2. Time/Energy (using padded)
    rms = librosa.feature.rms(y=y_padded)[0]
    features['rms_mean'] = np.mean(rms)
    features['rms_std'] = np.std(rms)
    features['rms_min'] = np.min(rms)
    features['rms_max'] = np.max(rms)

    zcr = librosa.feature.zero_crossing_rate(y_padded)[0]
    features['zcr_mean'] = np.mean(zcr)
    features['zcr_std'] = np.std(zcr)

    # 3. Spectral Ext (using padded)
    cent = librosa.feature.spectral_centroid(y=y_padded, sr=sr)[0]
    features['spectral_centroid_mean'] = np.mean(cent)
    features['spectral_centroid_std'] = np.std(cent)

    bw = librosa.feature.spectral_bandwidth(y=y_padded, sr=sr)[0]
    features['spectral_bandwidth_mean'] = np.mean(bw)
    features['spectral_bandwidth_std'] = np.std(bw)

    rolloff = librosa.feature.spectral_rolloff(y=y_padded, sr=sr)[0]
    features['spectral_rolloff_mean'] = np.mean(rolloff)
    features['spectral_rolloff_std'] = np.std(rolloff)
    
    if 'S' not in locals():
        S, _ = librosa.magphase(librosa.stft(y_padded))
    flux = np.mean(np.diff(S, axis=1)**2, axis=0)
    features['spectral_flux_mean'] = np.mean(flux) if len(flux) > 0 else 0.0

    # 4. MFCC
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
        
    return features


# Let's test on audio samples
import pandas as pd
df = pd.read_csv('data/splits_multilingual/train.csv')
test_df = pd.read_csv('data/splits_multilingual/test.csv')
full_df = pd.concat([df, test_df])
# sample a few normal ones
normal_ids = full_df[full_df['language'].isin(['Tamil', 'English', 'Hindi'])].head(20)['id'].tolist()
failed_ids = ['ASM_M_INDIC_00295', 'BEN_F_NAMES_01441', 'BEN_M_SURPRISE_00464', 'BRX_M_WIKI_01181', 'NEP_F_NEWS_00633', 'TAM_F_HAPPY_00096']

import datasets
from datasets import load_dataset
ds = load_dataset('parquet', data_files=glob.glob(os.path.expanduser('~/.cache/huggingface/hub/datasets--SherryT997--IndicTTS-Deepfake-Challenge-Data/snapshots/*/data/train-*.parquet')))

def get_audio(audio_id):
    for split in ds.keys():
        for item in ds[split]:
            if item['id'] == audio_id:
                return item['audio']['array'], item['audio']['sampling_rate']
    return None, None

results = []
# test 20 normals
for aid in normal_ids:
    arr, sr = get_audio(aid)
    if arr is not None:
        v2 = extract_features_v2(arr, sr)
        v3 = extract_features_v3(arr, sr)
        for k in v2:
            diff = abs(v2[k] - v3[k])
            if diff > 1e-4:
                results.append({'id': aid, 'feature': k, 'v2': v2[k], 'v3': v3[k], 'diff': diff})

print(f"Regression diffs: {len(results)}")
if len(results) > 0:
    print("Example diffs:", results[:5])

print("Testing failed IDs with V3:")
for aid in failed_ids:
    arr, sr = get_audio(aid)
    if arr is not None:
        try:
            v3 = extract_features_v3(arr, sr)
            print(f"{aid} - SUCCESS, Length: {len(arr)}, Dims: {len(v3)}, sum: {sum(v3.values())}")
        except Exception as e:
            print(f"{aid} - FAILED: {str(e)}")

