import sys
import os

# --- THE ULTIMATE MONKEY-PATCH MAC FIX ---
# Prevents torchaudio from crashing on Mac by intercepting the load function
import torch
import torchaudio
import librosa

def safe_audio_load(filepath, *args, **kwargs):
    data, sr = librosa.load(filepath, sr=None, mono=False)
    tensor = torch.tensor(data, dtype=torch.float32)
    if tensor.ndim == 1:
        tensor = tensor.unsqueeze(0)
    return tensor, sr

torchaudio.load = safe_audio_load
os.environ["TORCHAUDIO_USE_TORCHCODEC"] = "0"
sys.modules['torchcodec'] = None
# -----------------------------------------

import glob
import time
import torch.nn.functional as F
from msclap import CLAP
from transformers import WhisperModel, WhisperFeatureExtractor

# --- Configuration ---
DATASET_DIR = "./IITMandi_YouTube" 

LANG_MAP = {
    "asm": "Assamese", "ben": "Bengali", "bho": "Bhojpuri", "dog": "Dogri",
    "guj": "Gujarati", "hin": "Hindi", "kan": "Kannada", "kas": "Kashmiri",
    "kok": "Konkani", "mai": "Maithili", "mal": "Malayalam", "mar": "Marathi",
    "nep": "Nepali", "odi": "Odia", "pan": "Punjabi", "san": "Santali",
    "snd": "Sindhi", "tam": "Tamil", "tel": "Telugu", "urd": "Urdu",
    "bhi": "Bhili", "gon": "Gondi"
}

INDIAN_LANGUAGES = list(LANG_MAP.values())

# --- Setup Device & Models ---
use_gpu = torch.cuda.is_available()
device = torch.device('cuda' if use_gpu else 'cpu')
print(f"Loading Models on: {device}...")

# 1. Load Microsoft CLAP purely for the TEXT ENCODER
print("Loading Microsoft CLAP Text Encoder...")
clap_model = CLAP(version='2023', use_cuda=use_gpu)

# 2. Load Whisper purely for the AUDIO ENCODER
print("Loading Whisper Audio Encoder...")
whisper_feature_extractor = WhisperFeatureExtractor.from_pretrained("openai/whisper-base")
whisper_model = WhisperModel.from_pretrained("openai/whisper-base").to(device)
whisper_model.eval()

# --- Pre-compute Text Embeddings (Using MS-CLAP) ---
text_prompts = [f"A person speaking in the {lang} language." for lang in INDIAN_LANGUAGES]
print("Generating Microsoft CLAP text embeddings...")
with torch.no_grad():
    text_embeddings = clap_model.get_text_embeddings(text_prompts)
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

print(f"Starting MSCLAP + Whisper Hybrid Zero-Shot Evaluation on {len(audio_files)} files...")

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
            # 3. Load Audio (Whisper STRICTLY requires 16000 Hz)
            waveform, sr = librosa.load(audio_path, sr=16000)
            
            # 4. Extract features for Whisper
            inputs = whisper_feature_extractor(waveform, sampling_rate=16000, return_tensors="pt")
            input_features = inputs.input_features.to(device)
            
            # 5. Pass through Whisper's Audio Encoder
            encoder_outputs = whisper_model.encoder(input_features)
            audio_embedding = encoder_outputs.last_hidden_state.mean(dim=1) # Mean pooling
            
            # --- DIMENSION MATCHING FIX ---
            # MS-CLAP 2023 text is 1024D, Whisper-base audio is 512D.
            # We pad the Whisper vector with zeros so the cosine similarity math doesn't crash.
            if audio_embedding.shape[1] != text_embeddings.shape[1]:
                target_dim = text_embeddings.shape[1]
                if audio_embedding.shape[1] < target_dim:
                    pad_size = target_dim - audio_embedding.shape[1]
                    audio_embedding = F.pad(audio_embedding, (0, pad_size))
                else:
                    audio_embedding = audio_embedding[:, :target_dim]
            # ------------------------------

            # 6. Compute Cosine Similarity between Whisper Audio and MS-CLAP Text
            similarities = F.cosine_similarity(audio_embedding, text_embeddings)
            
            # 7. Predict
            predicted_index = torch.argmax(similarities).item()
            predicted_label = INDIAN_LANGUAGES[predicted_index]
            
            total_evaluated += 1
            if predicted_label == true_label:
                correct_predictions += 1
                
        except Exception as e:
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
    print("🏆 FINAL MSCLAP + WHISPER ZERO-SHOT RESULTS")
    print("="*45)
    print(f"Total Files Evaluated : {total_evaluated}")
    print(f"Correct Predictions   : {correct_predictions}")
    print(f"Final Accuracy        : {final_accuracy:.2f}%")
    print(f"Total Time Taken      : {total_time_str}")
    print("="*45)
else:
    print("\nEvaluation finished, but no valid files were processed.")