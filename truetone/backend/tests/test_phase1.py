import pytest
import numpy as np
import asyncio
import os
from app.preprocessing.audio import preprocess_audio
from app.simulator.simulator import CallSimulator

def test_preprocessing():
    # Generate 1 second of random noise
    sr = 44100
    audio = np.random.uniform(-1.0, 1.0, sr).astype(np.float32)
    
    # Process
    processed_audio = preprocess_audio(audio, sr)
    
    # Check shape and sample rate assumptions
    # It should be 1D, and depending on VAD, length should be <= original duration in 16kHz
    assert len(processed_audio.shape) == 1
    # Check max value is around 1.0 due to normalization (unless it's empty)
    if len(processed_audio) > 0:
        assert np.max(np.abs(processed_audio)) > 0.0

@pytest.mark.asyncio
async def test_simulator():
    # Make sure sample files exist (created by our script)
    genuine_path = "../sample_audio/genuine_sample.wav"
    cloned_path = "../sample_audio/cloned_sample.wav"
    
    assert os.path.exists(genuine_path)
    assert os.path.exists(cloned_path)
    
    sim = CallSimulator(window_duration_sec=3.0, step_duration_sec=1.0)
    
    # Test genuine stream
    call_id = "test-call-1"
    chunks = []
    
    # We only take the first 3 chunks to avoid waiting too long in the test
    stream = sim.stream_audio_file(genuine_path, call_id, "A->B", "genuine")
    async for chunk in stream:
        chunks.append(chunk)
        if len(chunks) == 3:
            break
            
    assert len(chunks) == 3
    assert chunks[0]["call_id"] == call_id
    assert chunks[0]["label"] == "genuine"
    assert "audio_data" in chunks[0]
    assert chunks[0]["sr"] == 16000 # Since we generated it at 16kHz
    
    # Verify window lengths
    expected_samples = 3.0 * 16000
    assert len(chunks[0]["audio_data"]) == int(expected_samples)
    assert len(chunks[1]["audio_data"]) == int(expected_samples)
    
    # Check preprocessing on one chunk
    processed = preprocess_audio(chunks[0]["audio_data"], chunks[0]["sr"])
    assert len(processed.shape) == 1
