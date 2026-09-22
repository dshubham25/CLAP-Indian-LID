import os
import soundfile as sf
from datasets import load_dataset
from tqdm import tqdm

# All 12 South Asian / Indian languages available in Google FLEURS (100% public, no login required)
FLEURS_LANGS = {
    "hi_in": "Hindi",
    "bn_in": "Bengali",
    "mr_in": "Marathi",
    "ta_in": "Tamil",
    "te_in": "Telugu",
    "gu_in": "Gujarati",
    "ur_pk": "Urdu",
    "pa_in": "Punjabi",
    "ml_in": "Malayalam",
    "as_in": "Assamese",
    "or_in": "Odia",
    "kn_in": "Kannada"
}

OUTPUT_DIR = os.path.expanduser("~/clap_fusion/fleurs_extracted")

def download_fleurs():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print("=" * 60)
    print("DOWNLOADING GOOGLE FLEURS (Indian Languages)")
    print(f"Target Directory: {OUTPUT_DIR}")
    print("=" * 60)
    
    for lang_code, lang_name in FLEURS_LANGS.items():
        lang_dir = os.path.join(OUTPUT_DIR, lang_name)
        if os.path.exists(lang_dir) and len(os.listdir(lang_dir)) >= 200:
            print(f"[SKIP] {lang_name} already has {len(os.listdir(lang_dir))} files.")
            continue
            
        print(f"\n[STREAMING] Downloading 200 samples for {lang_name} ({lang_code})...")
        os.makedirs(lang_dir, exist_ok=True)
        
        try:
            # Streams directly from Hugging Face - zero auth or disk bloat
            dataset = load_dataset("google/fleurs", lang_code, split="test", streaming=True)
            
            count = 0
            for item in tqdm(dataset, total=200, desc=lang_name):
                if count >= 200:
                    break
                    
                audio = item['audio']
                audio_array = audio['array']
                sample_rate = audio['sampling_rate']
                
                out_path = os.path.join(lang_dir, f"{lang_name}_{count}.wav")
                sf.write(out_path, audio_array, sample_rate)
                count += 1
                
            print(f"  Successfully saved {count} files for {lang_name}")
            
        except Exception as e:
            print(f"  Error downloading {lang_name}: {e}")

if __name__ == "__main__":
    download_fleurs()
