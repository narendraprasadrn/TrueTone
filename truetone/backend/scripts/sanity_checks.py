import numpy as np
import librosa
import soundfile as sf
import os
import sys

base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(base_dir)

from app.preprocessing.pipeline import preprocess_for_detection, prepare_aasist_context
from app.detection.aasist import AasistDetector

detector = AasistDetector(os.path.join(base_dir, "third_party/aasist/models/weights/AASIST-L.pth"))

def log_stats(name, window):
    print(f"[{name}] dtype: {window.dtype}, shape: {window.shape}, min: {window.min():.4f}, max: {window.max():.4f}, mean: {window.mean():.6f}")
    if window.dtype != np.float32 or window.max() > 1.01 or window.min() < -1.01 or window.shape != (64600,):
        print(f"  --> WARNING: {name} might be incorrect.")

def run_check_1():
    print("\n--- Check 1: Upload types ---")
    os.makedirs("debug", exist_ok=True)
    t = np.linspace(0, 5, 5*16000, endpoint=False)
    audio = (np.sin(2*np.pi*440*t) * 0.8).astype(np.float32)
    
    sf.write("debug/test.wav", audio, 16000)
    sf.write("debug/test.flac", audio, 16000, format='FLAC')
    sf.write("debug/test.ogg", audio, 16000, format='OGG')
    
    for ext in ["wav", "flac", "ogg"]:
        y, sr = librosa.load(f"debug/test.{ext}", sr=None)
        proc = preprocess_for_detection(y, sr)
        ctx = prepare_aasist_context(proc)
        log_stats(f"Upload .{ext}", ctx)
        
    # Live mic path uses np.frombuffer float32
    y = audio.tobytes()
    live = np.frombuffer(y, dtype=np.float32)
    proc = preprocess_for_detection(live, 16000)
    ctx = prepare_aasist_context(proc)
    log_stats("Live Mic", ctx)

def run_check_5():
    print("\n--- Check 5: Synthetic Inputs ---")
    sr = 16000
    t = np.linspace(0, 4.0375, int(sr*4.0375), endpoint=False)
    
    silence = np.zeros_like(t, dtype=np.float32)
    noise = np.random.randn(len(t)).astype(np.float32) * (10 ** (-40 / 20))
    tone = (np.sin(2 * np.pi * 440 * t) * 0.5).astype(np.float32)
    
    for name, y in [("Silence", silence), ("Noise -40dBFS", noise), ("Tone 440Hz", tone)]:
        proc = preprocess_for_detection(y, sr)
        ctx = prepare_aasist_context(proc)
        res = detector.predict_aasist(ctx, 16000)
        print(f"{name}: spoof_score={res['spoof_score']:.4f}, logits={res['logits']}")

def run_check_6():
    print("\n--- Check 6: Scale Invariance ---")
    sr = 16000
    t = np.linspace(0, 4.0375, int(sr*4.0375), endpoint=False)
    # create complex signal
    sig = (np.sin(2 * np.pi * 440 * t) * 0.2 + np.random.randn(len(t))*0.05).astype(np.float32)
    
    proc1 = preprocess_for_detection(sig, sr)
    ctx1 = prepare_aasist_context(proc1)
    res1 = detector.predict_aasist(ctx1, 16000)
    
    proc01 = preprocess_for_detection(sig * 0.1, sr)
    ctx01 = prepare_aasist_context(proc01)
    res01 = detector.predict_aasist(ctx01, 16000)
    
    print(f"Gain x1.0: spoof_score={res1['spoof_score']:.4f}")
    print(f"Gain x0.1: spoof_score={res01['spoof_score']:.4f}")
    diff = abs(res1['spoof_score'] - res01['spoof_score'])
    print(f"Difference: {diff:.4f}")
    if diff > 0.05:
         print("  --> WARNING: large difference, scale variance problem!")

if __name__ == "__main__":
    run_check_1()
    run_check_5()
    run_check_6()
