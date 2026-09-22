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
from prompts_llm_enhanced import LLM_ENHANCED_PROMPTS

# Layout mapping for the ekstep_seen dataset folders
FOLDER_TO_LANG = {
    "hin": "Hindi", "ben": "Bengali", "mar": "Marathi", "tel": "Telugu",
    "tam": "Tamil", "guj": "Gujarati", "urd": "Urdu", "kan": "Kannada",
    "ori": "Odia", "odi": "Odia", "mal": "Malayalam", "pun": "Punjabi",
    "asm": "Assamese"
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

def get_seen_unseen_split():
    all_langs = ["Hindi", "Bengali", "Marathi", "Telugu", "Tamil", "Gujarati", 
                 "Urdu", "Kannada", "Odia", "Malayalam", "Punjabi", "Assamese"]
    
    np.random.seed(42)
    shuffled_langs = np.random.permutation(all_langs)
    
    num_seen = int(len(all_langs) * 0.7)
    seen_langs = shuffled_langs[:num_seen].tolist()
    unseen_langs = shuffled_langs[num_seen:].tolist()
    
    return seen_langs, unseen_langs

def run_enhanced_evaluation(dataset_root, results_file="zero_shot_llm_enhanced_results.txt"):
    start_time = time.time()
    
    # Initialize MS-CLAP
    clap_model = CLAP(version='2023', use_cuda=torch.cuda.is_available())
    
    # INJECT THE PATCH INTO MS-CLAP
    clap_model.read_audio = types.MethodType(custom_read_audio, clap_model)
    
    wav_files = glob.glob(os.path.join(dataset_root, "**", "*.wav"), recursive=True)
    if not wav_files:
        print(f"No .wav files found in {dataset_root}. Check your path structure.")
        return
        
    seen_langs, _ = get_seen_unseen_split()
    
    # Filter files to only those in seen_langs
    files_to_process = []
    for f in wav_files:
        folder = os.path.basename(os.path.dirname(f))
        if folder in FOLDER_TO_LANG and FOLDER_TO_LANG[folder] in seen_langs:
            files_to_process.append(f)
            
    target_classes = seen_langs
    
    print(f"Total .wav files found in {dataset_root}: {len(wav_files)}")
    print(f"Files to process (seen languages only): {len(files_to_process)}\n")
    
    if len(files_to_process) == 0:
        print("No files to process!")
        return

    # Define simple text prompts
    simple_prompts = [f"A person speaking in {lang}" for lang in target_classes]
    simple_text_embeddings = clap_model.get_text_embeddings(simple_prompts)
    
    # Define enhanced text prompts (ensembled)
    enhanced_embeddings_list = []
    for lang in target_classes:
        if lang in LLM_ENHANCED_PROMPTS:
            prompts = LLM_ENHANCED_PROMPTS[lang]
            embeddings = clap_model.get_text_embeddings(prompts)
            mean_embedding = torch.mean(embeddings, dim=0, keepdim=True)
            enhanced_embeddings_list.append(mean_embedding)
        else:
            prompt = [f"A person speaking in {lang}"]
            embedding = clap_model.get_text_embeddings(prompt)
            enhanced_embeddings_list.append(embedding)
    enhanced_text_embeddings = torch.cat(enhanced_embeddings_list, dim=0)
    
    y_true = []
    y_pred_simple = []
    y_pred_enhanced = []
    
    with open(results_file, "w") as f:
        f.write("File | True Lang | Simple Pred | Enhanced Pred | Simple Match | Enhanced Match\n")
    
    # Process files with a progress bar
    for file_path in tqdm(files_to_process, desc="Evaluating"):
        folder = os.path.basename(os.path.dirname(file_path))
        true_lang = FOLDER_TO_LANG[folder]
        true_idx = target_classes.index(true_lang)
        
        try:
            # Extract audio features
            audio_embeddings = clap_model.get_audio_embeddings([file_path], resample=True)
            
            # Simple predictions
            sim_simple = clap_model.compute_similarity(audio_embeddings, simple_text_embeddings)
            pred_idx_simple = np.argmax(F.softmax(sim_simple.detach().cpu(), dim=1).numpy(), axis=1)[0]
            pred_lang_simple = target_classes[pred_idx_simple]
            
            # Enhanced predictions
            sim_enhanced = clap_model.compute_similarity(audio_embeddings, enhanced_text_embeddings)
            pred_idx_enhanced = np.argmax(F.softmax(sim_enhanced.detach().cpu(), dim=1).numpy(), axis=1)[0]
            pred_lang_enhanced = target_classes[pred_idx_enhanced]
            
            # Record predictions
            y_true.append(true_idx)
            y_pred_simple.append(pred_idx_simple)
            y_pred_enhanced.append(pred_idx_enhanced)
            
            match_simple = "MATCH" if true_lang == pred_lang_simple else "MISMATCH"
            match_enhanced = "MATCH" if true_lang == pred_lang_enhanced else "MISMATCH"
            
            # Write immediately to file
            result_str = f"{file_path} | {true_lang} | {pred_lang_simple} | {pred_lang_enhanced} | {match_simple} | {match_enhanced}\n"
            with open(results_file, "a") as f:
                f.write(result_str)
                
        except Exception as e:
            with open(results_file, "a") as f:
                f.write(f"{file_path} | ERROR: {str(e)}\n")

    if len(y_true) == 0:
        print("\nAll files threw an error.")
        return

    # Calculate overall metrics
    end_time = time.time()
    elapsed_time = end_time - start_time
    hours, rem = divmod(elapsed_time, 3600)
    minutes, seconds = divmod(rem, 60)
    
    acc_simple = accuracy_score(y_true, y_pred_simple)
    acc_enhanced = accuracy_score(y_true, y_pred_enhanced)
    
    summary_text = (
        f"\n--- Final Summary ---\n"
        f"Processed {len(y_true)} files successfully.\n"
        f"Time Taken: {int(hours)}h {int(minutes)}m {seconds:.2f}s\n"
        f"Simple Prompts Accuracy: {acc_simple * 100:.2f}%\n"
        f"Enhanced Prompts Accuracy: {acc_enhanced * 100:.2f}%\n"
        f"Improvement: {(acc_enhanced - acc_simple) * 100:.2f}%\n"
        f"\n--- Per-Language Accuracy Breakdown ---\n"
    )
    
    for idx, lang in enumerate(target_classes):
        lang_indices = [i for i, t in enumerate(y_true) if t == idx]
        if lang_indices:
            lang_acc_simple = accuracy_score([y_true[i] for i in lang_indices], [y_pred_simple[i] for i in lang_indices])
            lang_acc_enhanced = accuracy_score([y_true[i] for i in lang_indices], [y_pred_enhanced[i] for i in lang_indices])
            summary_text += f"{lang}: Simple = {lang_acc_simple*100:.2f}%, Enhanced = {lang_acc_enhanced*100:.2f}%\n"
    
    print(summary_text)
    
    # Append the final summary to the bottom of the text file
    with open(results_file, "a") as f:
        f.write(summary_text)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default='ekstep_seen')
    parser.add_argument('--results-file', default='zero_shot_llm_enhanced_results.txt')
    args = parser.parse_args()

    run_enhanced_evaluation(args.dataset, results_file=args.results_file)
