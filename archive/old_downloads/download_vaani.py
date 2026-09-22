"""
Downloads the Indian language subset of Vaani from HuggingFace.
Vaani is a large-scale Indian language dataset from IISc/Google.
"""
import os
import soundfile as sf
import numpy as np
from datasets import load_dataset

# Vaani language codes (IISc Bangalore / Google dataset)
VAANI_INDIAN_LANGS = {
    "Hindi": "Hindi",
    "Bengali": "Bengali",
    "Gujarati": "Gujarati",
    "Kannada": "Kannada",
    "Malayalam": "Malayalam",
    "Marathi": "Marathi",
    "Odia": "Odia",
    "Punjabi": "Punjabi",
    "Tamil": "Tamil",
    "Telugu": "Telugu",
    "Urdu": "Urdu",
    "Assamese": "Assamese",
    "English": "English",
}

OUTPUT_DIR = "Vaani_Indian"
MAX_PER_LANG = 200

def download_and_save():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Try multiple possible HuggingFace dataset names for Vaani
    possible_names = [
        "ai4bharat/vaani",
        "google/vaani", 
        "IISc/vaani",
    ]
    
    # First, try to find the correct dataset name
    dataset_name = None
    for name in possible_names:
        try:
            print(f"Trying to load dataset: {name}...")
            ds = load_dataset(name, split="test", streaming=True, trust_remote_code=True)
            # If we get here without error, this is the right name
            dataset_name = name
            print(f"  Found dataset: {name}")
            break
        except Exception as e:
            print(f"  Not found: {e}")
            continue
    
    if dataset_name is None:
        print("\n" + "="*60)
        print("VAANI DATASET NOT FOUND ON HUGGINGFACE")
        print("="*60)
        print("\nThe Vaani dataset may require manual download from:")
        print("  https://vaani.iisc.ac.in/")
        print("  or through the AI4Bharat portal")
        print("\nIf you have access, download the audio files and place them in:")
        print(f"  {OUTPUT_DIR}/<language_name_lowercase>/")
        print("\nExample structure:")
        print(f"  {OUTPUT_DIR}/hindi/*.wav")
        print(f"  {OUTPUT_DIR}/bengali/*.wav")
        print(f"  {OUTPUT_DIR}/tamil/*.wav")
        print("\nThen run: python evaluate_dataset.py --dataset Vaani_Indian")
        return
    
    # If found, download
    for lang_code, lang_name in VAANI_INDIAN_LANGS.items():
        print(f"\nDownloading: {lang_name}...")
        lang_dir = os.path.join(OUTPUT_DIR, lang_name.lower())
        os.makedirs(lang_dir, exist_ok=True)
        
        try:
            ds = load_dataset(dataset_name, lang_code, split="test", 
                            streaming=True, trust_remote_code=True)
            count = 0
            for i, sample in enumerate(ds):
                if count >= MAX_PER_LANG:
                    break
                audio = sample["audio"]
                audio_array = np.array(audio["array"], dtype=np.float32)
                sr = audio["sampling_rate"]
                out_path = os.path.join(lang_dir, f"vaani_{lang_code}_{i:04d}.wav")
                sf.write(out_path, audio_array, sr)
                count += 1
            print(f"  Saved {count} files")
        except Exception as e:
            print(f"  ERROR: {e}")

if __name__ == "__main__":
    download_and_save()
