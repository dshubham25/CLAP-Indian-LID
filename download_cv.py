import os
import soundfile as sf
from datasets import load_dataset
from tqdm import tqdm

# We will download 200 samples of these languages to test
CV_LANGS = {
    "hi": "Hindi", "bn": "Bengali", "mr": "Marathi", "ta": "Tamil",
    "te": "Telugu", "gu": "Gujarati", "ur": "Urdu", "pa": "Punjabi",
    "ml": "Malayalam", "as": "Assamese", "or": "Odia"
}

OUTPUT_DIR = os.path.expanduser("~/clap_fusion/common_voice_extracted")

def download_common_voice():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    for lang_code, lang_name in CV_LANGS.items():
        lang_dir = os.path.join(OUTPUT_DIR, lang_name)
        if os.path.exists(lang_dir) and len(os.listdir(lang_dir)) >= 200:
            print(f"Skipping {lang_name} (already downloaded)")
            continue
            
        print(f"\nDownloading Common Voice for {lang_name} ({lang_code})...")
        os.makedirs(lang_dir, exist_ok=True)
        
        try:
            # stream=True prevents downloading the entire massive dataset
            dataset = load_dataset("mozilla-foundation/common_voice_17_0", lang_code, split="test", streaming=True, trust_remote_code=True)
            
            count = 0
            for item in dataset:
                if count >= 200:
                    break
                    
                audio = item['audio']
                audio_array = audio['array']
                sample_rate = audio['sampling_rate']
                
                out_path = os.path.join(lang_dir, f"{lang_name}_{count}.wav")
                sf.write(out_path, audio_array, sample_rate)
                count += 1
                
            print(f"Successfully downloaded {count} files for {lang_name}")
            
        except Exception as e:
            print(f"Could not download {lang_name}: {e}")
            print("Note: If you get a 401 Unauthorized error, you must login to Hugging Face via `huggingface-cli login` and accept the terms at https://huggingface.co/datasets/mozilla-foundation/common_voice_17_0")

if __name__ == "__main__":
    download_common_voice()
