import pyarrow.parquet as pq
import pyarrow.compute as pc
import soundfile as sf
import io
import librosa
import glob
import os
import numpy as np

def load_audio_fast(needed_ids):
    cache_dir = os.path.expanduser("~/.cache/huggingface/hub/datasets--SherryT997--IndicTTS-Deepfake-Challenge-Data/snapshots/*/data/train-*.parquet")
    all_files = glob.glob(cache_dir)
    audio_cache = {}
    needed_ids_set = set(needed_ids)
    
    for f in all_files:
        table = pq.read_table(f, columns=["id", "audio"])
        ids = table.column("id").to_pylist()
        
        # find indices
        indices = [i for i, idx in enumerate(ids) if idx in needed_ids_set]
        if not indices:
            continue
            
        audio_col = table.column("audio")
        
        for i in indices:
            idx = ids[i]
            item = audio_col[i].as_py()
            if 'bytes' in item and item['bytes'] is not None:
                audio_bytes = item['bytes']
                data, sr = sf.read(io.BytesIO(audio_bytes))
                if len(data.shape) > 1: data = data.mean(axis=1)
                if sr != 16000:
                    data = librosa.resample(data, orig_sr=sr, target_sr=16000)
                audio_cache[idx] = data
            elif 'array' in item and item['array'] is not None:
                audio_cache[idx] = np.array(item['array'])
                
            if len(audio_cache) == len(needed_ids_set):
                return audio_cache
                
    return audio_cache
