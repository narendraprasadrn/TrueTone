import os
from datasets import load_dataset

def main():
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    cache_dir = os.path.join(base_dir, "data", "indictts")
    os.makedirs(cache_dir, exist_ok=True)
    
    print(f"Downloading dataset to {cache_dir}...")
    # This will download the entire dataset and save it to the specified cache directory
    ds = load_dataset("SherryT997/IndicTTS-Deepfake-Challenge-Data", cache_dir=cache_dir)
    print("Download complete.")
    print(ds)

if __name__ == "__main__":
    main()
