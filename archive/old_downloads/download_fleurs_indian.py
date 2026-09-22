"""
Downloads the Indian language subset of FLEURS from HuggingFace
and organizes it into a folder structure compatible with our pipeline.
"""
import os
import soundfile as sf
import numpy as np
from datasets import load_dataset

# All Indian languages available in FLEURS
FLEURS_INDIAN_LANGS = {
    "as_in": "Assamese",
    "bn_in": "Bengali", 
    "gu_in": "Gujarati",
    "hi_in": "Hindi",
    "kn_in": "Kannada",
    "ml_in": "Malayalam",
    "mr_in": "Marathi",
    "ne_np": "Nepali",
    "or_in": "Odia",
    "pa_in": "Punjabi",
    "sd_in": "Sindhi",
    "ta_in": "Tamil",
    "te_in": "Telugu",
    "ur_pk": "Urdu",
}

OUTPUT_DIR = "FLEURS_Indian"
MAX_PER_LANG = 200  # Keep it manageable for feature extraction

def download_and_save():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    for lang_code, lang_name in FLEURS_INDIAN_LANGS.items():
        print(f"\n{'='*50}")
        print(f"Downloading: {lang_name} ({lang_code})")
        print(f"{'='*50}")
        
        lang_dir = os.path.join(OUTPUT_DIR, lang_name.lower())
        os.makedirs(lang_dir, exist_ok=True)
        
        try:
            # Load the test split from FLEURS
            ds = load_dataset("google/fleurs", lang_code, split="test", trust_remote_code=True)
            
            count = 0
            for i, sample in enumerate(ds):
                if count >= MAX_PER_LANG:
                    break
                    
                audio = sample["audio"]
                audio_array = np.array(audio["array"], dtype=np.float32)
                sr = audio["sampling_rate"]
                
                out_path = os.path.join(lang_dir, f"{lang_code}_{i:04d}.wav")
                sf.write(out_path, audio_array, sr)
                count += 1
                
            print(f"  Saved {count} files to {lang_dir}")
            
        except Exception as e:
            print(f"  ERROR downloading {lang_name}: {e}")
    
    # Print summary
    print(f"\n{'='*50}")
    print("DOWNLOAD COMPLETE")
    print(f"{'='*50}")
    total = 0
    for lang_name in FLEURS_INDIAN_LANGS.values():
        lang_dir = os.path.join(OUTPUT_DIR, lang_name.lower())
        if os.path.exists(lang_dir):
            n = len([f for f in os.listdir(lang_dir) if f.endswith('.wav')])
            total += n
            print(f"  {lang_name}: {n} files")
    print(f"\nTotal: {total} files across {len(FLEURS_INDIAN_LANGS)} languages")
    print(f"Saved to: {OUTPUT_DIR}/")

if __name__ == "__main__":
    download_and_save()
