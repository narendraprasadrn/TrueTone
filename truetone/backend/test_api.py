import requests
import json

with open("real_whatsapp_uncompressed.wav", "rb") as f:
    res = requests.post("http://localhost:8000/test/analyze-audio", files={"file": f})

if res.status_code == 200:
    data = res.json()
    print(f"Windows count: {len(data['windows'])}")
    for i, w in enumerate(data['windows']):
        print(f"W{i}: AASIST={w['aasist_score']:.2f}, Prosody={w['prosody_score']:.2f}, Fused={w['fused_score']:.2f}")
    
    print("\nBefore (Last Window):")
    lw = data['windows'][-1]
    print(f"AASIST: {lw['aasist_score']*100:.0f}%, Prosody: {lw['prosody_score']*100:.0f}%, Fused: {lw['fused_score']*100:.0f}%, Class: {data['overall_classification']}")
    
    print("\nAfter (Consolidated):")
    aasist = sum(w['aasist_score'] for w in data['windows'] if w['aasist_score'] is not None) / len([w for w in data['windows'] if w['aasist_score'] is not None])
    prosody = sum(w['prosody_score'] for w in data['windows'] if w['prosody_score'] is not None) / len([w for w in data['windows'] if w['prosody_score'] is not None])
    fused = sum(w['fused_score'] for w in data['windows'] if w['fused_score'] is not None) / len([w for w in data['windows'] if w['fused_score'] is not None])
    cls = "HIGH" if fused > 0.69 else "MEDIUM" if fused > 0.39 else "LOW"
    print(f"AASIST: {aasist*100:.0f}%, Prosody: {prosody*100:.0f}%, Fused: {fused*100:.0f}%, Class: {cls}")
else:
    print(f"Failed: {res.text}")
