import os
import sys
import asyncio
import numpy as np
import soundfile as sf
from pathlib import Path
from unittest.mock import patch

# Add backend to sys.path
backend_dir = Path(__file__).resolve().parent.parent
if str(backend_dir) not in sys.path:
    sys.path.append(str(backend_dir))

from app.detection.aasist import AasistDetector
from app.detection.prosody import ProsodyDetector
from app.detection.speaker_verification import SpeakerVerifier
from app.simulator.simulator import CallSimulator

async def main():
    print("Loading models...")
    
    aasist = AasistDetector(str(backend_dir / "third_party" / "aasist" / "models" / "weights" / "AASIST-L.pth"))
    
    # We will use the fallback 0.5 prosody score if lgbm model is not trained
    prosody = ProsodyDetector(str(backend_dir / "app" / "detection" / "models" / "prosody_lgbm.txt"))
    
    sv = SpeakerVerifier(savedir=str(backend_dir / "pretrained_models" / "ecapa-tdnn"))
    
    print("Enrolling speaker with genuine sample (first 5 seconds)...")
    gen_path = str(backend_dir.parent / "sample_audio" / "genuine_sample.wav")
    cln_path = str(backend_dir.parent / "sample_audio" / "cloned_sample.wav")
    
    gen_audio, sr = sf.read(gen_path)
    enroll_audio = gen_audio[:int(sr * 5.0)]
    sv.enroll("user_123", enroll_audio)
    print("Enrollment complete.")
    
    sim = CallSimulator(window_duration_sec=4.0, step_duration_sec=1.0)
    
    print("\n--- Testing Genuine Sample ---")
    print(f"{'window (s)':<15} | {'aasist_score':<15} | {'prosody_score':<15} | {'speaker_score':<15}")
    print("-" * 70)
    
    # Monkeypatch asyncio.sleep so we don't wait in real time for tests
    with patch("asyncio.sleep", return_value=None):
        async for frame in sim.stream_audio_file(gen_path, "call_1", "inbound", "genuine"):
            audio = frame["audio_data"]
            win_str = f"{frame['window_start_sec']:.1f}-{frame['window_end_sec']:.1f}"
            
            a_score = aasist.score(audio, sr)
            p_score = prosody.score(audio, sr)
            s_score = sv.score(audio, "user_123")
            s_score_str = f"{s_score:.4f}" if s_score is not None else "N/A"
            
            print(f"{win_str:<15} | {a_score:<15.4f} | {p_score:<15.4f} | {s_score_str:<15}")

    print("\n--- Testing Cloned Sample ---")
    print(f"{'window (s)':<15} | {'aasist_score':<15} | {'prosody_score':<15} | {'speaker_score':<15}")
    print("-" * 70)
    
    with patch("asyncio.sleep", return_value=None):
        async for frame in sim.stream_audio_file(cln_path, "call_2", "inbound", "cloned"):
            audio = frame["audio_data"]
            win_str = f"{frame['window_start_sec']:.1f}-{frame['window_end_sec']:.1f}"
            
            a_score = aasist.score(audio, sr)
            p_score = prosody.score(audio, sr)
            s_score = sv.score(audio, "user_123")
            s_score_str = f"{s_score:.4f}" if s_score is not None else "N/A"
            
            print(f"{win_str:<15} | {a_score:<15.4f} | {p_score:<15.4f} | {s_score_str:<15}")

if __name__ == "__main__":
    asyncio.run(main())
