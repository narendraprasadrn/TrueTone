import datasets
ds = datasets.load_dataset("SpeechAntiSpoofingBenchmarks/ASVspoof2019_LA", split="test", streaming=True).cast_column("audio", datasets.Audio(decode=False))
for i, ex in enumerate(ds):
    print(f"Sample {i}: label={ex.get('label')}, is_spoof={ex.get('is_spoof')}, id={ex.get('id', ex.get('audio', {}).get('path'))}")
    if i > 5: break
