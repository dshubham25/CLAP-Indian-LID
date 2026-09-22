import os
import glob
import time
import types
import random
import torch
import soundfile as sf
import torchaudio.transforms as T
import numpy as np
import torch.nn.functional as F
from tqdm import tqdm
from sklearn.metrics import accuracy_score
from msclap import CLAP

FOLDER_TO_LANG = {
    "hin": "Hindi", "ben": "Bengali", "mar": "Marathi", "tel": "Telugu",
    "tam": "Tamil", "guj": "Gujarati", "urd": "Urdu", "kan": "Kannada",
    "ori": "Odia", "odi": "Odia", "mal": "Malayalam", "pun": "Punjabi",
    "asm": "Assamese", "mai": "Maithili", "sat": "Santali", "kas": "Kashmiri",
    "nep": "Nepali", "snd": "Sindhi", "kok": "Konkani", "doi": "Dogri",
    "mni": "Manipuri", "brx": "Bodo", "san": "Sanskrit", "eng": "English",
}

# --- THE MONKEY PATCH (Mac Fix) ---
def custom_read_audio(self, audio_path, resample=True):
    audio_array, sample_rate = sf.read(audio_path)
    if len(audio_array.shape) == 1:
        audio_tensor = torch.FloatTensor(audio_array).unsqueeze(0)
    else:
        audio_tensor = torch.FloatTensor(audio_array).T
    resample_rate = self.args.sampling_rate
    if resample and resample_rate != sample_rate:
        resampler = T.Resample(sample_rate, resample_rate)
        audio_tensor = resampler(audio_tensor)
    return audio_tensor, resample_rate

def get_processed_files(results_file):
    processed = set()
    if os.path.exists(results_file):
        with open(results_file, "r") as f:
            for line in f:
                if "|" in line:
                    file_path = line.split("|")[0].strip()
                    processed.add(file_path)
    return processed

def run_zero_shot_language_split(dataset_root, results_file="zero_shot_lang_split_results.txt"):
    clap_model = CLAP(version='2023', use_cuda=False)
    clap_model.read_audio = types.MethodType(custom_read_audio, clap_model)
    
    wav_files = glob.glob(os.path.join(dataset_root, "**", "*.wav"), recursive=True)
    if not wav_files:
        print(f"No .wav files found in {dataset_root}.")
        return

    # Determine languages present
    present_folders = list(set([os.path.basename(os.path.dirname(f)) for f in wav_files if os.path.basename(os.path.dirname(f)) in FOLDER_TO_LANG]))
    all_langs = sorted(list(set([FOLDER_TO_LANG[f] for f in present_folders])))
    
    # 70/30 Language Split
    random.seed(42)
    random.shuffle(all_langs)
    split_idx = int(0.7 * len(all_langs))
    seen_langs = all_langs[:split_idx]
    unseen_langs = all_langs[split_idx:]
    
    # Check progress
    processed_files = get_processed_files(results_file)
    files_to_process = [f for f in wav_files if f not in processed_files]
    
    print(f"\n--- Zero-Shot Language Split ---")
    print(f"Seen Languages (70%): {seen_langs}")
    print(f"Unseen Languages (30%): {unseen_langs}")
    print(f"Files remaining to process: {len(files_to_process)}\n")

    if len(files_to_process) > 0:
        # Pre-compute text embeddings for ALL languages
        prompts = [f"A person speaking in {lang}" for lang in all_langs]
        text_embeddings = clap_model.get_text_embeddings(prompts)
        
        for file_path in tqdm(files_to_process, desc="Evaluating"):
            folder = os.path.basename(os.path.dirname(file_path))
            if folder not in FOLDER_TO_LANG:
                continue
            
            true_lang = FOLDER_TO_LANG[folder]
            group = "SEEN" if true_lang in seen_langs else "UNSEEN"
            
            try:
                audio_embeddings = clap_model.get_audio_embeddings([file_path], resample=True)
                similarity = clap_model.compute_similarity(audio_embeddings, text_embeddings)
                
                pred_idx = np.argmax(F.softmax(similarity.detach().cpu(), dim=1).numpy(), axis=1)[0]
                pred_lang = all_langs[pred_idx]
                match_status = "MATCH" if true_lang == pred_lang else "MISMATCH"
                
                with open(results_file, "a") as f:
                    f.write(f"{file_path} | Group: {group} | True: {true_lang} | Pred: {pred_lang} | {match_status}\n")
            except Exception as e:
                with open(results_file, "a") as f:
                    f.write(f"{file_path} | ERROR: {str(e)}\n")

    # Tally up the final results from the text file
    print("\nProcessing complete. Tallying results...")
    y_true_seen, y_pred_seen = [], []
    y_true_unseen, y_pred_unseen = [], []
    
    with open(results_file, "r") as f:
        for line in f:
            if "Group:" in line and "ERROR" not in line:
                parts = [p.strip() for p in line.split("|")]
                group = parts[1].split(":")[1].strip()
                true_lang = parts[2].split(":")[1].strip()
                pred_lang = parts[3].split(":")[1].strip()
                
                if group == "SEEN":
                    y_true_seen.append(true_lang)
                    y_pred_seen.append(pred_lang)
                else:
                    y_true_unseen.append(true_lang)
                    y_pred_unseen.append(pred_lang)
                    
    acc_seen = accuracy_score(y_true_seen, y_pred_seen) if y_true_seen else 0
    acc_unseen = accuracy_score(y_true_unseen, y_pred_unseen) if y_true_unseen else 0
    
    print(f"\nZero-Shot Accuracy on SEEN Languages: {acc_seen * 100:.2f}%")
    print(f"Zero-Shot Accuracy on UNSEEN Languages: {acc_unseen * 100:.2f}%")

if __name__ == "__main__":
    DATASET_PATH = "ekstep_seen"
    run_zero_shot_language_split(DATASET_PATH)