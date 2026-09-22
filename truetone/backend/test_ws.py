import asyncio
import websockets
import numpy as np

async def test():
    async with websockets.connect("ws://localhost:8000/ws/live-mic") as ws:
        print("Sending noise...")
        for _ in range(50): # send 50 chunks of 4096 = 204800 samples (enough for 3 windows)
            chunk = np.random.randn(4096).astype(np.float32) * 0.1
            await ws.send(chunk.tobytes())
            try:
                res = await asyncio.wait_for(ws.recv(), timeout=0.1)
                print("Noise score:", res)
            except asyncio.TimeoutError:
                pass
asyncio.run(test())
