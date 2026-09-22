import asyncio
import soundfile as sf
import time
import uuid

class CallSimulator:
    def __init__(self, window_duration_sec: float = 2.0, step_duration_sec: float = 1.0):
        self.window_duration_sec = window_duration_sec
        self.step_duration_sec = step_duration_sec

    async def stream_audio_file(
        self, 
        filepath: str, 
        call_id: str, 
        direction: str, 
        label: str,
        first_time_caller: bool = False,
        transcript_stub: str = ""
    ):
        """
        Reads an audio file and yields rolling windows of audio data 
        at approximately real playback pace.
        """
        try:
            audio_data, sr = sf.read(filepath)
        except Exception as e:
            print(f"Error reading {filepath}: {e}")
            return
            
        # Ensure mono
        if len(audio_data.shape) > 1:
            audio_data = audio_data[:, 0]

        window_samples = int(self.window_duration_sec * sr)
        step_samples = int(self.step_duration_sec * sr)
        
        total_samples = len(audio_data)
        start_idx = 0
        
        # Simulate real-time streaming
        start_time = time.time()
        
        while start_idx + window_samples <= total_samples:
            end_idx = start_idx + window_samples
            window_chunk = audio_data[start_idx:end_idx]
            
            # Calculate time we should wait to match "real playback pace"
            # For the first chunk, we wait until window_duration_sec has elapsed since start.
            # Then we emit every step_duration_sec.
            # But we can also simulate it by just sleeping step_duration_sec per iteration
            # if we assume the first chunk takes window_duration_sec to buffer.
            
            yield {
                "call_id": call_id,
                "direction": direction,
                "label": label,
                "audio_data": window_chunk,
                "sr": sr,
                "timestamp": time.time(),
                "window_start_sec": start_idx / sr,
                "window_end_sec": end_idx / sr,
                "first_time_caller": first_time_caller,
                "transcript_stub": transcript_stub
            }
            
            await asyncio.sleep(self.step_duration_sec)
            start_idx += step_samples

        # Yield any remaining part if it's significant
        if start_idx < total_samples and (total_samples - start_idx) > sr * 1.0: # At least 1 second
            window_chunk = audio_data[start_idx:]
            yield {
                "call_id": call_id,
                "direction": direction,
                "label": label,
                "audio_data": window_chunk,
                "sr": sr,
                "timestamp": time.time(),
                "window_start_sec": start_idx / sr,
                "window_end_sec": total_samples / sr,
                "first_time_caller": first_time_caller,
                "transcript_stub": transcript_stub
            }

