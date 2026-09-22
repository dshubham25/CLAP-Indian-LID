import sys
import os

import glob
import time
import torch
import torchaudio
import librosa

def safe_audio_load(filepath, *args, **kwargs):
    # Load audio safely with librosa (bypassing torchaudio's C++ backends entirely)
    data, sr = librosa.load(filepath, sr=None, mono=False)
    tensor = torch.tensor(data, dtype=torch.float32)
    # Ensure it has the (channels, time) shape PyTorch expects
    if tensor.ndim == 1:
        tensor = tensor.unsqueeze(0)
    return tensor, sr

# 💥 OVERWRITE the broken PyTorch function with our safe version
torchaudio.load = safe_audio_load

import torch.nn.functional as F
from msclap import CLAP

# --- Configuration ---
DATASET_DIR = "./IITMandi_YouTube" 

# Mapping the 3-letter folder/file codes to the full language names
LANG_MAP = {
    "asm": "Assamese", "ben": "Bengali", "bho": "Bhojpuri", "dog": "Dogri",
    "guj": "Gujarati", "hin": "Hindi", "kan": "Kannada", "kas": "Kashmiri",
    "kok": "Konkani", "mai": "Maithili", "mal": "Malayalam", "mar": "Marathi",
    "nep": "Nepali", "odi": "Odia", "pan": "Punjabi", "san": "Santali",
    "snd": "Sindhi", "tam": "Tamil", "tel": "Telugu", "urd": "Urdu",
    "bhi": "Bhili", "gon": "Gondi"
}

INDIAN_LANGUAGES = list(LANG_MAP.values())

# --- Setup Device & Model ---
use_gpu = torch.cuda.is_available()
device = torch.device('cuda' if use_gpu else 'cpu')
print(f"Loading Microsoft CLAP model on: {device}...")

# Load the official Microsoft CLAP model (ICASSP 2023 version)
# This automatically handles the text (RoBERTa) and audio encoders internally
clap_model = CLAP(version='2023', use_cuda=use_gpu)

# --- Pre-compute Text Embeddings ---
# Convert the language list into semantic text prompts
text_prompts = [f"A person speaking in the {lang} language." for lang in INDIAN_LANGUAGES]

print("Generating Microsoft text embeddings...")
with torch.no_grad():
    # msclap returns a tensor directly
    text_embeddings = clap_model.get_text_embeddings(text_prompts)
    # Ensure it's on the correct device for similarity math
    text_embeddings = text_embeddings.to(device)

# --- Evaluation Loop ---
print(f"Scanning directory for audio files: {DATASET_DIR}")
audio_files = glob.glob(os.path.join(DATASET_DIR, "**", "*.wav"), recursive=True)

if not audio_files:
    print(f"ERROR: No .wav files found in {DATASET_DIR}. Please check your path.")
    exit()

correct_predictions = 0
total_evaluated = 0
start_time = time.time()

print(f"Starting Microsoft CLAP Zero-Shot Evaluation on {len(audio_files)} files...")

with torch.no_grad():
    for idx, audio_path in enumerate(audio_files):
        # 1. Dynamically extract the 3-letter code
        folder_code = os.path.basename(os.path.dirname(audio_path)).lower()
        if folder_code not in LANG_MAP:
            filename = os.path.basename(audio_path).lower()
            folder_code = filename[:3]
            
        # 2. Skip if not in our target list
        if folder_code not in LANG_MAP:
            continue
            
        true_label = LANG_MAP[folder_code]
            
        try:
            # 3. Extract audio embedding using Microsoft's native pipeline
            audio_embedding = clap_model.get_audio_embeddings([audio_path])
            audio_embedding = audio_embedding.to(device)
            
            # 4. Compute Cosine Similarity
            similarities = F.cosine_similarity(audio_embedding, text_embeddings)
            
            # 5. Predict
            predicted_index = torch.argmax(similarities).item()
            predicted_label = INDIAN_LANGUAGES[predicted_index]
            
            # 6. Tally results
            total_evaluated += 1
            if predicted_label == true_label:
                correct_predictions += 1
                
        except Exception as e:
            # Safety net for corrupted .wav files
            print(f"Error processing {audio_path}: {e}")
            continue
            
        # Print progress update and ETA every 100 files
        if total_evaluated > 0 and total_evaluated % 100 == 0:
            current_acc = (correct_predictions / total_evaluated) * 100
            
            elapsed_time = time.time() - start_time
            time_per_file = elapsed_time / total_evaluated
            eta_seconds = (len(audio_files) - total_evaluated) * time_per_file
            
            elapsed_str = time.strftime('%H:%M:%S', time.gmtime(elapsed_time))
            eta_str = time.strftime('%H:%M:%S', time.gmtime(eta_seconds))
            
            print(f"Processed {total_evaluated}/{len(audio_files)}... Acc: {current_acc:.2f}% | Elapsed: {elapsed_str} | ETA: {eta_str}")

# --- Final Results ---
if total_evaluated > 0:
    final_accuracy = (correct_predictions / total_evaluated) * 100
    total_time = time.time() - start_time
    total_time_str = time.strftime('%H:%M:%S', time.gmtime(total_time))
    
    print("\n" + "="*45)
    print("CLAP ZERO-SHOT RESULTS")
    print("="*45)
    print(f"Total Files Evaluated : {total_evaluated}")
    print(f"Correct Predictions   : {correct_predictions}")
    print(f"Final Accuracy        : {final_accuracy:.2f}%")
    print(f"Total Time Taken      : {total_time_str}")
    print("="*45)
else:
    print("\nEvaluation finished, but no valid files were processed.")