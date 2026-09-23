import asyncio
import soundfile as sf
import time
import uuid

class CallSimulator:
    def __init__(self, window_duration_sec: float = 2.0, step_duration_sec: float = 1.0, history_sec: float = 4.04):
        self.window_duration_sec = window_duration_sec
        self.step_duration_sec = step_duration_sec
        self.history_sec = history_sec

    async def stream_audio_file(
        self, 
        filepath: str, 
        call_id: str, 
        direction: str, 
        label: str,
        first_time_caller: bool = False,
        transcript_stub: str = ""
    ):
        try:
            audio_data, sr = sf.read(filepath)
        except Exception as e:
            print(f"Error reading {filepath}: {e}")
            return
            
        if len(audio_data.shape) > 1:
            audio_data = audio_data[:, 0]

        step_samples = int(self.step_duration_sec * sr)
        history_samples = int(self.history_sec * sr)
        total_samples = len(audio_data)
        
        # Start at 2.0s
        current_end_sec = 2.0
        start_time = time.time()
        
        while current_end_sec * sr <= total_samples:
            end_idx = int(current_end_sec * sr)
            start_idx = max(0, end_idx - history_samples)
            
            window_chunk = audio_data[start_idx:end_idx]
            
            yield {
                "call_id": call_id,
                "direction": direction,
                "label": label,
                "audio_data": window_chunk,  # This now contains up to 4.04s history
                "sr": sr,
                "timestamp": time.time(),
                "window_start_sec": max(0.0, current_end_sec - self.window_duration_sec),
                "window_end_sec": current_end_sec,
                "first_time_caller": first_time_caller,
                "transcript_stub": transcript_stub
            }
            
            await asyncio.sleep(self.step_duration_sec)
            current_end_sec += self.step_duration_sec

        if current_end_sec * sr > total_samples and total_samples / sr - (current_end_sec - self.step_duration_sec) > 0.5:
            end_idx = total_samples
            start_idx = max(0, end_idx - history_samples)
            window_chunk = audio_data[start_idx:end_idx]
            yield {
                "call_id": call_id,
                "direction": direction,
                "label": label,
                "audio_data": window_chunk,
                "sr": sr,
                "timestamp": time.time(),
                "window_start_sec": max(0.0, total_samples / sr - self.window_duration_sec),
                "window_end_sec": total_samples / sr,
                "first_time_caller": first_time_caller,
                "transcript_stub": transcript_stub
            }
