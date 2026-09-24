import asyncio
import websockets
import numpy as np
import librosa
import soundfile as sf
import os
import time

async def send_audio(name, audio):
    print(f"\n--- Testing: {name} ---")
    uri = "ws://localhost:8000/ws/live-mic"
    results = []
    try:
        async with websockets.connect(uri) as ws:
            chunk_size = 16000
            for i in range(0, len(audio), chunk_size):
                chunk = audio[i:i+chunk_size]
                if len(chunk) < chunk_size: break
                await ws.send(chunk.tobytes())
                try:
                    res = await asyncio.wait_for(ws.recv(), timeout=2.0)
                    results.append(res)
                except asyncio.TimeoutError:
                    pass
    except Exception as e:
        print(e)
    for r in results:
        print(r)

async def main():
    sr = 16000
    dur = 6.0
    t = np.linspace(0, dur, int(sr*dur), endpoint=False)
    
    # 1. Fan on, silent room (Noise + 50Hz)
    fan = np.random.randn(len(t)).astype(np.float32) * 0.01 + np.sin(2*np.pi*50*t)*0.015
    fan = fan.astype(np.float32)
    
    # 2. Fan on, me speaking normally
    try:
        speech, _ = librosa.load("natural.wav", sr=sr)
        if len(speech) > len(t): speech = speech[:len(t)]
        else: speech = np.pad(speech, (0, len(t)-len(speech)))
    except:
        speech = np.sin(2*np.pi*440*t).astype(np.float32) # fallback
        
    fan_speech = (fan + speech).astype(np.float32)
    
    # 3. Fan off, me speaking normally
    quiet_speech = speech.astype(np.float32)
    
    await send_audio("Fan ON, Silent Room", fan)
    await send_audio("Fan ON, Speaking", fan_speech)
    await send_audio("Fan OFF, Speaking", quiet_speech)

if __name__ == "__main__":
    asyncio.run(main())
