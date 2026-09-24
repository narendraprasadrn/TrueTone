import asyncio
import websockets
import numpy as np
import json
import os
import requests

def create_audio():
    sr = 16000
    duration = 5 
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)
    
    silence = np.zeros_like(t, dtype=np.float32)
    
    # White noise with peak ~0.015
    noise = np.random.randn(len(t)).astype(np.float32)
    noise = noise / np.max(np.abs(noise)) * 0.015 
    
    # Room tone with peak ~0.02
    room = noise + np.sin(2 * np.pi * 50 * t) * 0.005
    room = room.astype(np.float32)

    return silence, noise, room

async def test_live_mic(audio, name):
    print(f"\n--- Testing Live Mic: {name} ---")
    uri = "ws://localhost:8000/ws/live-mic"
    try:
        async with websockets.connect(uri) as websocket:
            chunk_size = 16000
            for i in range(0, len(audio), chunk_size):
                chunk = audio[i:i+chunk_size]
                await websocket.send(chunk.tobytes())
                try:
                    res = await asyncio.wait_for(websocket.recv(), timeout=2.0)
                    print(f"WS Result: {res}")
                except asyncio.TimeoutError:
                    pass
    except Exception as e:
        print(f"WS Error: {e}")

def test_test_bench(audio, name):
    print(f"\n--- Testing Test Bench: {name} ---")
    import scipy.io.wavfile as wav
    os.makedirs("debug", exist_ok=True)
    path = f"debug/temp_{name}.wav"
    wav.write(path, 16000, audio)
    
    with open(path, "rb") as f:
        res = requests.post("http://localhost:8000/test/analyze-audio", files={"file": f})
        
    if res.status_code == 200:
        data = res.json()
        for w in data.get("windows", []):
            print(f"Window {w['window_index']}: AASIST={w['aasist_score']:.4f}, Prosody={w['prosody_score']:.4f}, Fused={w['fused_score']:.4f}, Class={w['classification']}")
    else:
        print(f"HTTP Error: {res.status_code} - {res.text}")

async def main():
    silence, noise, room = create_audio()
    for audio, name in [(silence, "silence"), (noise, "noise"), (room, "room")]:
        await test_live_mic(audio, name)
        test_test_bench(audio, name)

if __name__ == "__main__":
    asyncio.run(main())
