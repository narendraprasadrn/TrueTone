import json

families = {
    "PROSODIC": ["f0_mean", "f0_std", "f0_range", "f0_median", "f0_min", "f0_max", "voiced_ratio"],
    "VOICE QUALITY": ["jitter", "shimmer", "hnr"],
    "BEHAVIOURAL/TEMPORAL": ["pause_count", "pause_mean_dur", "pause_max_dur", "pause_median_dur", "speech_duration", "silence_duration", "speech_silence_ratio", "speech_segment_count", "temporal_regularity"],
    "ACOUSTIC": ["flatness_mean", "flatness_std", "rms_mean", "rms_std", "rms_min", "rms_max", "zcr_mean", "zcr_std", "spectral_centroid_mean", "spectral_centroid_std", "spectral_bandwidth_mean", "spectral_bandwidth_std", "spectral_rolloff_mean", "spectral_rolloff_std", "spectral_flux_mean"]
}

for i in range(13):
    families["ACOUSTIC"].append(f"mfcc_{i}_mean")
    families["ACOUSTIC"].append(f"mfcc_{i}_std")
    families["ACOUSTIC"].append(f"mfcc_delta_{i}_mean")
    families["ACOUSTIC"].append(f"mfcc_delta_{i}_std")
    families["ACOUSTIC"].append(f"mfcc_delta2_{i}_mean")
    families["ACOUSTIC"].append(f"mfcc_delta2_{i}_std")

all_features = []
# V2 order:
v2_order = ['f0_mean', 'f0_std', 'f0_range', 'f0_median', 'f0_min', 'f0_max', 'jitter', 'shimmer', 'hnr', 'voiced_ratio',
            'pause_count', 'pause_mean_dur', 'pause_max_dur', 'speech_duration', 'silence_duration', 'speech_silence_ratio',
            'flatness_mean', 'flatness_std', 'rms_mean', 'rms_std', 'rms_min', 'rms_max', 'zcr_mean', 'zcr_std',
            'spectral_centroid_mean', 'spectral_centroid_std', 'spectral_bandwidth_mean', 'spectral_bandwidth_std',
            'spectral_rolloff_mean', 'spectral_rolloff_std', 'spectral_flux_mean']

for i in range(13):
    v2_order.extend([f"mfcc_{i}_mean", f"mfcc_{i}_std", f"mfcc_delta_{i}_mean", f"mfcc_delta_{i}_std", f"mfcc_delta2_{i}_mean", f"mfcc_delta2_{i}_std"])

v3_order = v2_order.copy()
# Insert new features where appropriate
# Insert pause_median_dur after pause_max_dur
v3_order.insert(v3_order.index('pause_max_dur') + 1, 'pause_median_dur')
# Insert speech_segment_count and temporal_regularity after speech_silence_ratio
v3_order.insert(v3_order.index('speech_silence_ratio') + 1, 'speech_segment_count')
v3_order.insert(v3_order.index('speech_segment_count') + 1, 'temporal_regularity')

def get_fam(f):
    for k,v in families.items():
        if f in v: return k
    return "UNKNOWN"

records = []
for idx, f in enumerate(v3_order):
    records.append({
        "index": idx,
        "name": f,
        "family": get_fam(f),
        "description": f.replace("_", " ").title()
    })

with open("feature_names_v3.json", "w") as f:
    json.dump(records, f, indent=2)

print(f"Generated {len(records)} feature names.")
