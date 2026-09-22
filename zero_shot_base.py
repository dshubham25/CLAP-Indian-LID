import os
import glob
import time
import types
import torch
import soundfile as sf
import torchaudio.transforms as T
import numpy as np
import torch.nn.functional as F
from tqdm import tqdm
from sklearn.metrics import accuracy_score
from msclap import CLAP
import argparse
from prompts import EKSTEP_PROMPTS

# Layout mapping for the ekstep_seen dataset folders
FOLDER_TO_LANG = {
    "hin": "Hindi", "ben": "Bengali", "mar": "Marathi", "tel": "Telugu",
    "tam": "Tamil", "guj": "Gujarati", "urd": "Urdu", "kan": "Kannada",
    "ori": "Odia", "odi": "Odia", "mal": "Malayalam", "pun": "Punjabi",
    "asm": "Assamese", "mai": "Maithili", "sat": "Santali", "kas": "Kashmiri",
    "nep": "Nepali", "snd": "Sindhi", "kok": "Konkani", "doi": "Dogri",
    "mni": "Manipuri", "brx": "Bodo", "san": "Sanskrit", "eng": "English",
}

# --- THE MONKEY PATCH ---
# This bypasses the broken torchaudio FFmpeg backend on Mac by forcing msclap to use soundfile directly.
def custom_read_audio(self, audio_path, resample=True):
    audio_array, sample_rate = sf.read(audio_path)
    
    # Convert numpy array to torch tensor matching torchaudio's format (channels, frames)
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

def run_base_zero_shot(dataset_root, results_file="zero_shot_results_fixed.txt", prompt_type="rich"):
    start_time = time.time()
    
    # Initialize MS-CLAP
    clap_model = CLAP(version='2023', use_cuda=False) 
    
    # INJECT THE PATCH INTO MS-CLAP
    clap_model.read_audio = types.MethodType(custom_read_audio, clap_model)
    
    # Gather all audio files recursively
    wav_files = glob.glob(os.path.join(dataset_root, "**", "*.wav"), recursive=True)
    if not wav_files:
        print(f"No .wav files found in {dataset_root}. Check your path structure.")
        return
        
    # Check what has already been processed to allow safe resuming
    processed_files = get_processed_files(results_file)
    files_to_process = [f for f in wav_files if f not in processed_files]
    
    total_found = len(wav_files)
    total_remaining = len(files_to_process)
    
    print(f"\n--- Evaluation Setup ---")
    print(f"Total .wav files found: {total_found}")
    print(f"Files already processed: {len(processed_files)}")
    print(f"Files remaining in this run: {total_remaining}\n")
    
    if total_remaining == 0:
        print("All files have already been processed!")
        return

    # Dynamically extract target classes present in the local folders
    present_folders = list(set([os.path.basename(os.path.dirname(f)) for f in wav_files]))
    target_classes = [FOLDER_TO_LANG[f] for f in present_folders if f in FOLDER_TO_LANG]
    
    # Define text prompts
    if prompt_type == 'rich':
        text_embeddings_list = []
        for lang in target_classes:
            if lang in EKSTEP_PROMPTS:
                prompts = EKSTEP_PROMPTS[lang]
                embeddings = clap_model.get_text_embeddings(prompts)
                mean_embedding = torch.mean(embeddings, dim=0, keepdim=True)
                text_embeddings_list.append(mean_embedding)
            else:
                prompt = [f"A person speaking in {lang}"]
                embedding = clap_model.get_text_embeddings(prompt)
                text_embeddings_list.append(embedding)
        text_embeddings = torch.cat(text_embeddings_list, dim=0)
    else:
        prompts = [f"A person speaking in {lang}" for lang in target_classes]
        text_embeddings = clap_model.get_text_embeddings(prompts)
    
    y_true, y_preds = [], []
    correct_count = 0
    
    # Process files with a progress bar
    for i, file_path in enumerate(tqdm(files_to_process, desc="Evaluating")):
        folder = os.path.basename(os.path.dirname(file_path))
        if folder not in FOLDER_TO_LANG:
            continue
            
        true_lang = FOLDER_TO_LANG[folder]
        true_idx = target_classes.index(true_lang)
        
        try:
            # Extract audio features and calculate semantic text match
            audio_embeddings = clap_model.get_audio_embeddings([file_path], resample=True)
            similarity = clap_model.compute_similarity(audio_embeddings, text_embeddings)
            
            preds_softmax = F.softmax(similarity.detach().cpu(), dim=1).numpy()
            pred_idx = np.argmax(preds_softmax, axis=1)[0]
            pred_lang = target_classes[pred_idx]
            
            # Record predictions
            y_true.append(true_idx)
            y_preds.append(pred_idx)
            
            match_status = "MATCH" if true_lang == pred_lang else "MISMATCH"
            if match_status == "MATCH":
                correct_count += 1
                
            # Write immediately to file
            result_str = f"{file_path} | True: {true_lang} | Pred: {pred_lang} | {match_status}\n"
            with open(results_file, "a") as f:
                f.write(result_str)
                
        except Exception as e:
            # If a specific audio file is actually corrupted, log it and keep going
            with open(results_file, "a") as f:
                f.write(f"{file_path} | ERROR: {str(e)}\n")

    # Guard clause against empty arrays (just in case they all fail)
    if len(y_true) == 0:
        print("\nAll remaining files threw an error. Please check the results text file for the specific error.")
        return

    # Calculate overall metrics
    end_time = time.time()
    elapsed_time = end_time - start_time
    hours, rem = divmod(elapsed_time, 3600)
    minutes, seconds = divmod(rem, 60)
    
    acc = accuracy_score(y_true, y_preds)
    summary_text = (
        f"\n--- Final Summary ---\n"
        f"Processed {len(y_true)} files successfully in this session.\n"
        f"Time Taken: {int(hours)}h {int(minutes)}m {seconds:.2f}s\n"
        f"Session Accuracy: {acc * 100:.2f}%\n"
    )
    
    print(summary_text)
    
    # Append the final summary to the bottom of the text file
    with open(results_file, "a") as f:
        f.write(summary_text)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--prompts', choices=['simple', 'rich'], default='rich')
    parser.add_argument('--results-file', default='zero_shot_results_rich_prompts.txt')
    args = parser.parse_args()

    DATASET_PATH = "ekstep_seen"
    run_base_zero_shot(DATASET_PATH, results_file=args.results_file, prompt_type=args.prompts)