"""
Downloads the Indian language subset of VoxLingua107.
The dataset is hosted by TalTechNLP on HuggingFace.
"""
import os
import soundfile as sf
import numpy as np
from datasets import load_dataset

VOXLINGUA_INDIAN_LANGS = {
    "hi": "Hindi",
    "bn": "Bengali",
    "gu": "Gujarati",
    "kn": "Kannada",
    "ml": "Malayalam",
    "mr": "Marathi",
    "or": "Odia",
    "pa": "Punjabi",
    "ta": "Tamil",
    "te": "Telugu",
    "ur": "Urdu",
    "as": "Assamese",
    "ne": "Nepali",
    "sd": "Sindhi",
}

OUTPUT_DIR = "VoxLingua_Indian"
MAX_PER_LANG = 200

def download_and_save():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Try loading the full dataset first (it has a 'language' column)
    dataset_names_to_try = [
        "TalTechNLP/VoxLingua107",
        "SEACrowd/voxlingua",
    ]
    
    ds = None
    for ds_name in dataset_names_to_try:
        try:
            print(f"Trying to load: {ds_name} ...")
            ds = load_dataset(ds_name, split="validation", streaming=True)
            print(f"  SUCCESS: Using {ds_name}")
            
            # Peek at first sample to understand the schema
            sample = next(iter(ds))
            print(f"  Columns: {list(sample.keys())}")
            break
        except Exception as e:
            print(f"  Failed: {e}")
            try:
                # Some datasets only have a 'train' split
                ds = load_dataset(ds_name, split="train", streaming=True)
                print(f"  SUCCESS (train split): Using {ds_name}")
                sample = next(iter(ds))
                print(f"  Columns: {list(sample.keys())}")
                break
            except Exception as e2:
                print(f"  Also failed with train split: {e2}")
                ds = None
                continue
    
    if ds is None:
        print("\n" + "="*60)
        print("COULD NOT LOAD VOXLINGUA107 FROM HUGGINGFACE")
        print("="*60)
        print("\nVoxLingua107 can be downloaded manually from:")
        print("  http://bark.phon.ioc.ee/voxlingua107/")
        print("\nDownload the zip files for the Indian languages you need,")
        print("extract them, and place them in:")
        print(f"  {OUTPUT_DIR}/<language_name_lowercase>/")
        print("\nExample structure:")
        print(f"  {OUTPUT_DIR}/hindi/*.wav")
        print(f"  {OUTPUT_DIR}/bengali/*.wav")
        print("\nThen run: python evaluate_dataset.py --dataset VoxLingua_Indian")
        return
    
    # Process the streaming dataset
    lang_counts = {lang: 0 for lang in VOXLINGUA_INDIAN_LANGS.values()}
    
    for sample in ds:
        # Check if we have enough for all languages
        if all(c >= MAX_PER_LANG for c in lang_counts.values()):
            break
        
        # Get the language code from the sample
        lang_code = sample.get("language", sample.get("lang", sample.get("label", None)))
        
        if lang_code in VOXLINGUA_INDIAN_LANGS:
            lang_name = VOXLINGUA_INDIAN_LANGS[lang_code]
            
            if lang_counts[lang_name] >= MAX_PER_LANG:
                continue
            
            lang_dir = os.path.join(OUTPUT_DIR, lang_name.lower())
            os.makedirs(lang_dir, exist_ok=True)
            
            try:
                audio = sample["audio"]
                audio_array = np.array(audio["array"], dtype=np.float32)
                sr = audio["sampling_rate"]
                
                idx = lang_counts[lang_name]
                out_path = os.path.join(lang_dir, f"voxlingua_{lang_code}_{idx:04d}.wav")
                sf.write(out_path, audio_array, sr)
                lang_counts[lang_name] += 1
                
                if lang_counts[lang_name] % 50 == 0:
                    print(f"  {lang_name}: {lang_counts[lang_name]} files saved...")
            except Exception as e:
                pass
    
    print(f"\n{'='*50}")
    print("DOWNLOAD COMPLETE")
    print(f"{'='*50}")
    total = 0
    for lang_name in VOXLINGUA_INDIAN_LANGS.values():
        lang_dir = os.path.join(OUTPUT_DIR, lang_name.lower())
        if os.path.exists(lang_dir):
            n = len([f for f in os.listdir(lang_dir) if f.endswith('.wav')])
            total += n
            print(f"  {lang_name}: {n} files")
    print(f"\nTotal: {total} files saved to {OUTPUT_DIR}/")

if __name__ == "__main__":
    download_and_save()
