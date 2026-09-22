"""
Downloads the Indian language subset of Mozilla Common Voice from HuggingFace.
Common Voice contains crowd-sourced read speech.
"""
import os
import soundfile as sf
import numpy as np
from datasets import load_dataset

COMMONVOICE_INDIAN_LANGS = {
    "hi": "Hindi",
    "bn": "Bengali",
    "gu": "Gujarati",
    "kn": "Kannada",
    "ml": "Malayalam",
    "mr": "Marathi",
    "or": "Odia",
    "pa-IN": "Punjabi",
    "ta": "Tamil",
    "te": "Telugu",
    "ur": "Urdu",
    "as": "Assamese",
    "ne-NP": "Nepali",
    "sd": "Sindhi",
}

OUTPUT_DIR = "CommonVoice_Indian"
MAX_PER_LANG = 200

def download_and_save():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Try different Common Voice versions
    cv_versions = [
        "mozilla-foundation/common_voice_17_0",
        "mozilla-foundation/common_voice_16_1",
        "mozilla-foundation/common_voice_16_0",
        "mozilla-foundation/common_voice_13_0",
        "mozilla-foundation/common_voice_11_0",
    ]
    
    working_version = None
    for version in cv_versions:
        try:
            print(f"Trying: {version} ...")
            # Test with Hindi first
            ds = load_dataset(version, "hi", split="test", streaming=True)
            sample = next(iter(ds))
            print(f"  SUCCESS: Using {version}")
            print(f"  Columns: {list(sample.keys())}")
            working_version = version
            break
        except Exception as e:
            print(f"  Failed: {e}")
            continue
    
    if working_version is None:
        print("\nERROR: Could not find a working Common Voice version.")
        print("You may need to accept the license agreement on HuggingFace first.")
        print("Visit: https://huggingface.co/datasets/mozilla-foundation/common_voice_17_0")
        print("and click 'Agree and Access'.")
        print("\nThen login via: huggingface-cli login")
        return
    
    for lang_code, lang_name in COMMONVOICE_INDIAN_LANGS.items():
        print(f"\n{'='*50}")
        print(f"Downloading: {lang_name} ({lang_code})")
        print(f"{'='*50}")
        
        lang_dir = os.path.join(OUTPUT_DIR, lang_name.lower())
        os.makedirs(lang_dir, exist_ok=True)
        
        try:
            ds = load_dataset(working_version, lang_code, split="test", streaming=True)
            
            count = 0
            for i, sample in enumerate(ds):
                if count >= MAX_PER_LANG:
                    break
                
                audio = sample["audio"]
                audio_array = np.array(audio["array"], dtype=np.float32)
                sr = audio["sampling_rate"]
                
                out_path = os.path.join(lang_dir, f"cv_{lang_code}_{i:04d}.wav")
                sf.write(out_path, audio_array, sr)
                count += 1
                
            print(f"  Saved {count} files to {lang_dir}")
            
        except Exception as e:
            print(f"  ERROR downloading {lang_name}: {e}")
    
    print(f"\n{'='*50}")
    print("DOWNLOAD COMPLETE")
    print(f"{'='*50}")
    total = 0
    for lang_name in COMMONVOICE_INDIAN_LANGS.values():
        lang_dir = os.path.join(OUTPUT_DIR, lang_name.lower())
        if os.path.exists(lang_dir):
            n = len([f for f in os.listdir(lang_dir) if f.endswith('.wav')])
            total += n
            print(f"  {lang_name}: {n} files")
    print(f"\nTotal: {total} files saved to {OUTPUT_DIR}/")

if __name__ == "__main__":
    download_and_save()
