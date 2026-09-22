import os
import glob
import time
import random
import argparse
import numpy as np
from tqdm import tqdm
from sklearn.metrics import accuracy_score, confusion_matrix
import whisper
import torch

# Layout mapping for the ekstep_seen dataset folders
FOLDER_TO_LANG = {
    "hin": "Hindi", "ben": "Bengali", "mar": "Marathi", "tel": "Telugu",
    "tam": "Tamil", "guj": "Gujarati", "urd": "Urdu", "kan": "Kannada",
    "ori": "Odia", "odi": "Odia", "mal": "Malayalam", "pun": "Punjabi",
    "asm": "Assamese", "mai": "Maithili", "sat": "Santali", "kas": "Kashmiri",
    "nep": "Nepali", "snd": "Sindhi", "kok": "Konkani", "doi": "Dogri",
    "mni": "Manipuri", "brx": "Bodo", "san": "Sanskrit", "eng": "English",
}

# Whisper's ISO codes mapping to full language names
WHISPER_TO_LANG = {
    "hi": "Hindi", "bn": "Bengali", "mr": "Marathi", "te": "Telugu",
    "ta": "Tamil", "gu": "Gujarati", "ur": "Urdu", "kn": "Kannada",
    "or": "Odia", "ml": "Malayalam", "pa": "Punjabi", "as": "Assamese",
    "ne": "Nepali", "sd": "Sindhi", "sa": "Sanskrit", "en": "English",
}

def get_processed_files(results_file):
    """Reads the results file to return a set of already processed file paths."""
    processed = set()
    if os.path.exists(results_file):
        with open(results_file, "r") as f:
            for line in f:
                if "|" in line:
                    file_path = line.split("|")[0].strip()
                    processed.add(file_path)
    return processed

def run_whisper_lid(dataset_root, model_size="base"):
    results_file = f"whisper_{model_size}_lid_results.txt"
    start_time = time.time()
    
    # Support MPS device for Apple Silicon, otherwise CUDA or CPU
    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"Loading Whisper model '{model_size}' on {device}...")
    model = whisper.load_model(model_size, device=device)
    
    wav_files = glob.glob(os.path.join(dataset_root, "**", "*.wav"), recursive=True)
    if not wav_files:
        print(f"No .wav files found in {dataset_root}. Check your path structure.")
        return

    # Dynamically extract target classes present in the local folders
    present_folders = list(set([os.path.basename(os.path.dirname(f)) for f in wav_files if os.path.basename(os.path.dirname(f)) in FOLDER_TO_LANG]))
    all_langs = sorted(list(set([FOLDER_TO_LANG[f] for f in present_folders])))
    
    # 70/30 Language Split with seed 42 to match other scripts
    random.seed(42)
    langs_copy = all_langs.copy()
    random.shuffle(langs_copy)
    split_idx = int(0.7 * len(langs_copy))
    seen_langs = langs_copy[:split_idx]
    unseen_langs = langs_copy[split_idx:]
    
    # Identify which languages are supported by Whisper
    supported_langs = list(WHISPER_TO_LANG.values())
    unsupported_in_dataset = [lang for lang in all_langs if lang not in supported_langs]
    supported_in_dataset = [lang for lang in all_langs if lang in supported_langs]

    print(f"\n--- Whisper Language Support Summary ---")
    print(f"Supported in dataset: {supported_in_dataset}")
    print(f"NOT supported in dataset: {unsupported_in_dataset}")
    
    processed_files = get_processed_files(results_file)
    files_to_process = [f for f in wav_files if f not in processed_files]
    
    print(f"\n--- Evaluation Setup ---")
    print(f"Seen Languages (70%): {seen_langs}")
    print(f"Unseen Languages (30%): {unseen_langs}")
    print(f"Total files found: {len(wav_files)}")
    print(f"Files already processed: {len(processed_files)}")
    print(f"Files remaining: {len(files_to_process)}\n")
    
    if len(files_to_process) > 0:
        for file_path in tqdm(files_to_process, desc="Evaluating"):
            folder = os.path.basename(os.path.dirname(file_path))
            if folder not in FOLDER_TO_LANG:
                continue
            
            true_lang = FOLDER_TO_LANG[folder]
            group = "SEEN" if true_lang in seen_langs else "UNSEEN"
            
            try:
                # Load and pad/trim audio
                audio = whisper.load_audio(file_path)
                audio = whisper.pad_or_trim(audio)
                
                # Make log-Mel spectrogram and move to the model's device
                mel = whisper.log_mel_spectrogram(audio, n_mels=model.dims.n_mels).to(model.device)
                
                # Detect language
                _, probs = model.detect_language(mel)
                pred_iso = max(probs, key=probs.get)
                
                # Map Whisper's ISO code to full language name
                pred_lang = WHISPER_TO_LANG.get(pred_iso, f"Unknown ({pred_iso})")
                
                match_status = "MATCH" if true_lang == pred_lang else "MISMATCH"
                
                # Write result to file with resumption support
                with open(results_file, "a") as f:
                    f.write(f"{file_path} | Group: {group} | True: {true_lang} | Pred: {pred_lang} | {match_status}\n")
            except Exception as e:
                with open(results_file, "a") as f:
                    f.write(f"{file_path} | ERROR: {str(e)}\n")

    # Final Evaluation & Metrics
    print("\nProcessing complete. Tallying results...")
    y_true_seen, y_pred_seen = [], []
    y_true_unseen, y_pred_unseen = [], []
    y_true_all, y_pred_all = [], []
    
    per_lang_results = {lang: {"correct": 0, "total": 0} for lang in all_langs}
    
    if not os.path.exists(results_file):
        print("No results found to evaluate.")
        return

    with open(results_file, "r") as f:
        for line in f:
            if "Group:" in line and "ERROR" not in line:
                parts = [p.strip() for p in line.split("|")]
                group = parts[1].split(":")[1].strip()
                true_lang = parts[2].split(":")[1].strip()
                pred_lang = parts[3].split(":")[1].strip()
                
                if true_lang in per_lang_results:
                    per_lang_results[true_lang]["total"] += 1
                    if true_lang == pred_lang:
                        per_lang_results[true_lang]["correct"] += 1
                
                y_true_all.append(true_lang)
                y_pred_all.append(pred_lang)
                
                if group == "SEEN":
                    y_true_seen.append(true_lang)
                    y_pred_seen.append(pred_lang)
                else:
                    y_true_unseen.append(true_lang)
                    y_pred_unseen.append(pred_lang)
                    
    acc_seen = accuracy_score(y_true_seen, y_pred_seen) if y_true_seen else 0
    acc_unseen = accuracy_score(y_true_unseen, y_pred_unseen) if y_true_unseen else 0
    acc_overall = accuracy_score(y_true_all, y_pred_all) if y_true_all else 0
    
    print(f"\n--- Results Summary ---")
    print(f"Overall Accuracy: {acc_overall * 100:.2f}%")
    print(f"Seen Languages Accuracy: {acc_seen * 100:.2f}%")
    print(f"Unseen Languages Accuracy: {acc_unseen * 100:.2f}%")
    
    print("\n--- Per-Language Accuracy ---")
    for lang in all_langs:
        stats = per_lang_results[lang]
        if stats["total"] > 0:
            acc = stats["correct"] / stats["total"] * 100
            print(f"{lang:15s}: {acc:6.2f}% ({stats['correct']}/{stats['total']})")
        else:
            print(f"{lang:15s}: N/A (0 files)")
            
    print("\n--- Confusion Matrix ---")
    labels = sorted(list(set(y_true_all + y_pred_all)))
    if labels:
        cm = confusion_matrix(y_true_all, y_pred_all, labels=labels)
        
        # Simple text representation of confusion matrix
        header = f"{'True / Pred':>15} | " + " ".join([f"{str(l)[:5]:>5}" for l in labels])
        print(header)
        print("-" * len(header))
        for i, true_l in enumerate(labels):
            if true_l in y_true_all: # Only print rows for true labels that actually exist in the test set
                row = f"{str(true_l):>15} | " + " ".join([f"{cm[i, j]:>5}" for j in range(len(labels))])
                print(row)
            
    end_time = time.time()
    elapsed = end_time - start_time
    hours, rem = divmod(elapsed, 3600)
    minutes, seconds = divmod(rem, 60)
    print(f"\nTotal Session Time: {int(hours)}h {int(minutes)}m {seconds:.2f}s")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Whisper Language Identification Baseline")
    parser.add_argument("--dataset_root", type=str, default="ekstep_seen", help="Path to the dataset directory")
    parser.add_argument("--model_size", type=str, default="base", choices=["tiny", "base", "small", "medium", "large"], help="Whisper model size")
    
    args = parser.parse_args()
    run_whisper_lid(args.dataset_root, args.model_size)
