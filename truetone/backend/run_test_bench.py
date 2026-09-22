import requests

def run_file(filepath, label):
    print(f"\n--- Processing {label} ---")
    with open(filepath, 'rb') as f:
        res = requests.post("http://localhost:5000/test/analyze-audio", files={"file": f}, data={"enrolled_identity_id": "user_demo"})
    
    if res.status_code != 200:
        print(f"Error: {res.status_code} - {res.text}")
        return
        
    data = res.json()
    print(f"Overall Classification: {data['overall_classification']}")
    print(f"Overall Score (R_final of last window): {data['overall_score']:.3f}")
    print("\nWindow by Window:")
    for w in data["windows"]:
        # fused_score in the API response maps to R_final, we can calculate R_window from signals
        aasist = w['aasist_score']
        prosody = w['prosody_score']
        speaker = w['speaker_score'] if w['speaker_score'] is not None else 0.0
        mismatch = max(0.0, min(1.0, 1.0 - speaker))
        r_window = 0.6*aasist + 0.25*prosody + 0.15*mismatch
        r_final = w['fused_score']
        print(f"  Window {w['window_index']}: R_window={r_window:.3f}, R_final={r_final:.3f} -> {w['classification']}")

run_file("../sample_audio/genuine_sample.wav", "Genuine Sample")
run_file("../sample_audio/cloned_sample.wav", "Cloned Sample")
