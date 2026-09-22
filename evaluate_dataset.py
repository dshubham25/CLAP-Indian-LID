"""
Universal Evaluation Script for the Gated Fusion Model.
Tests the trained model on ANY dataset folder with the structure:
  <dataset_dir>/<language_name_lowercase>/*.wav

Usage:
  python evaluate_dataset.py --dataset FLEURS_Indian
  python evaluate_dataset.py --dataset VoxLingua_Indian
  python evaluate_dataset.py --dataset CommonVoice_Indian
  python evaluate_dataset.py --dataset Vaani_Indian
  python evaluate_dataset.py --dataset IITMandi_YouTube
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import types
import soundfile as sf
import torchaudio.transforms as T
import os
import numpy as np
import argparse
from tqdm import tqdm
from msclap import CLAP
from transformers import AutoFeatureExtractor, WhisperModel
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

DEVICE = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')

# The 8 languages the model was TRAINED on
SEEN_LANGS = {'Marathi', 'Kannada', 'English', 'Odia', 'Punjabi', 'Malayalam', 'Telugu', 'Gujarati'}

# --- GATED FUSION ARCHITECTURE ---
class CLAPWhisperGatedFusion(nn.Module):
    def __init__(self, clap_dim=1024, whisper_dim=512, out_dim=1024, dropout=0.2):
        super(CLAPWhisperGatedFusion, self).__init__()
        self.gate_proj = nn.Linear(whisper_dim, clap_dim)
        self.gate_activation = nn.Sigmoid()
        self.fc_fuse = nn.Linear(clap_dim + whisper_dim, out_dim)
        self.dropout = nn.Dropout(dropout)
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))

    def forward(self, clap_audio, whisper_audio):
        attention_mask = self.gate_activation(self.gate_proj(whisper_audio))
        filtered_clap = clap_audio * attention_mask
        x = torch.cat([filtered_clap, whisper_audio], dim=1)
        x = self.fc_fuse(x)
        x = self.dropout(x)
        x = F.normalize(x, p=2, dim=1)
        return x

# --- HELPERS ---
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

def load_audio_for_whisper(path, target_sr=16000):
    audio, sr = sf.read(path)
    if len(audio.shape) > 1:
        audio = audio.mean(axis=1)
    if sr != target_sr:
        resampler = T.Resample(sr, target_sr)
        audio = resampler(torch.FloatTensor(audio)).numpy()
    return audio

def discover_languages(dataset_dir):
    """Auto-discover languages from folder names."""
    langs = []
    for d in sorted(os.listdir(dataset_dir)):
        full_path = os.path.join(dataset_dir, d)
        if os.path.isdir(full_path) and not d.startswith('.'):
            # Capitalize first letter of each word
            lang_name = d.title()
            # Check if it has wav files
            wav_files = [f for f in os.listdir(full_path) if f.endswith('.wav')]
            if len(wav_files) > 0:
                langs.append(lang_name)
                print(f"  Found: {lang_name} ({len(wav_files)} files)")
    return langs

def extract_features(dataset_dir, langs, cache_name):
    """Extract CLAP + Whisper features."""
    if os.path.exists(cache_name):
        print(f"\nLoading cached features from {cache_name}...")
        data = np.load(cache_name, allow_pickle=True)
        return data['X_clap'], data['X_whisper'], data['y'], data['lang_names']
    
    print(f"\nExtracting features for {dataset_dir}...")
    
    clap_model = CLAP(version='2023', use_cuda=False)
    clap_model.read_audio = types.MethodType(custom_read_audio, clap_model)
    
    whisper_id = "openai/whisper-base"
    processor = AutoFeatureExtractor.from_pretrained(whisper_id)
    whisper_model = WhisperModel.from_pretrained(whisper_id).encoder.to(DEVICE)
    whisper_model.eval()
    
    lang_to_idx = {l: i for i, l in enumerate(langs)}
    
    files_to_process = []
    for lang_name in langs:
        lang_dir = os.path.join(dataset_dir, lang_name.lower())
        if not os.path.exists(lang_dir):
            continue
        for f in sorted(os.listdir(lang_dir)):
            if f.endswith('.wav'):
                files_to_process.append((os.path.join(lang_dir, f), lang_to_idx[lang_name], lang_name))
    
    print(f"Found {len(files_to_process)} files across {len(langs)} languages.")
    
    clap_features, whisper_features, labels, lang_names_list = [], [], [], []
    
    for path, label, lang_name in tqdm(files_to_process, desc="Extracting features"):
        try:
            c_emb = clap_model.get_audio_embeddings([path], resample=True).detach().cpu().numpy().squeeze()
            
            w_audio = load_audio_for_whisper(path)
            inputs = processor(w_audio, sampling_rate=16000, return_tensors="pt")
            w_input = inputs.input_features.to(DEVICE)
            with torch.no_grad():
                out = whisper_model(w_input)
                w_emb = out.last_hidden_state.mean(dim=1).squeeze().cpu().numpy()
            
            clap_features.append(c_emb)
            whisper_features.append(w_emb)
            labels.append(label)
            lang_names_list.append(lang_name)
        except Exception as e:
            pass
    
    X_clap = np.array(clap_features)
    X_whisper = np.array(whisper_features)
    y = np.array(labels)
    lang_names_arr = np.array(lang_names_list)
    
    np.savez(cache_name, X_clap=X_clap, X_whisper=X_whisper, y=y, lang_names=lang_names_arr)
    print(f"Saved features to {cache_name}")
    return X_clap, X_whisper, y, lang_names_arr

def evaluate(args):
    dataset_name = args.dataset
    dataset_dir = dataset_name
    
    print("="*60)
    print(f"EVALUATING GATED FUSION ON: {dataset_name}")
    print("="*60)
    
    # 1. Discover languages in the dataset folder
    print(f"\nDiscovering languages in {dataset_dir}/...")
    all_langs = discover_languages(dataset_dir)
    
    if len(all_langs) == 0:
        print(f"ERROR: No language folders with .wav files found in {dataset_dir}/")
        return
    
    # Separate into seen and unseen
    seen = [l for l in all_langs if l in SEEN_LANGS]
    unseen = [l for l in all_langs if l not in SEEN_LANGS]
    print(f"\n  SEEN languages (trained on EkStep):  {seen}")
    print(f"  UNSEEN languages (never trained on): {unseen}")
    
    # 2. Generate text embeddings for ALL discovered languages
    print(f"\nGenerating text embeddings for {len(all_langs)} languages...")
    clap_model = CLAP(version='2023', use_cuda=False)
    prompts = [f"A person speaking in {lang}" for lang in all_langs]
    text_embeddings = clap_model.get_text_embeddings(prompts).detach().to(DEVICE)
    
    # 3. Extract features
    cache_name = f"features_fusion_{dataset_name}.npz"
    X_clap, X_whisper, y, lang_names = extract_features(dataset_dir, all_langs, cache_name)
    
    # 4. Load trained model
    print("\nLoading trained Gated Fusion model (fusion_model_best.pt)...")
    model = CLAPWhisperGatedFusion(clap_dim=1024, whisper_dim=512, out_dim=1024).to(DEVICE)
    model.load_state_dict(torch.load("fusion_model_best.pt", map_location=DEVICE))
    model.eval()
    
    # 5. Run inference
    print("Running zero-shot inference...\n")
    all_preds = []
    batch_size = 256
    
    with torch.no_grad():
        for i in range(0, len(X_clap), batch_size):
            c_batch = torch.FloatTensor(X_clap[i:i+batch_size]).to(DEVICE)
            w_batch = torch.FloatTensor(X_whisper[i:i+batch_size]).to(DEVICE)
            fused_audio = model(c_batch, w_batch)
            logits = model.logit_scale.exp() * fused_audio @ text_embeddings.T
            preds = torch.argmax(logits, dim=1)
            all_preds.extend(preds.cpu().numpy())
    
    all_preds = np.array(all_preds)
    
    # 6. Calculate metrics
    seen_mask = np.array([lang_names[i] in SEEN_LANGS for i in range(len(lang_names))])
    unseen_mask = ~seen_mask
    
    overall_acc = accuracy_score(y, all_preds) * 100
    
    print(f"{'='*60}")
    print(f"RESULTS FOR: {dataset_name}")
    print(f"{'='*60}")
    print(f"\n  OVERALL ACCURACY ({len(all_langs)} languages): {overall_acc:.2f}%")
    
    if seen_mask.sum() > 0:
        seen_acc = accuracy_score(y[seen_mask], all_preds[seen_mask]) * 100
        print(f"\n  SEEN LANGUAGES ({len(seen)} langs): {seen_acc:.2f}%")
        seen_labels = sorted(set(y[seen_mask]))
        seen_names = [all_langs[i] for i in seen_labels]
        print(classification_report(y[seen_mask], all_preds[seen_mask],
              labels=seen_labels, target_names=seen_names, zero_division=0))
    
    if unseen_mask.sum() > 0:
        unseen_acc = accuracy_score(y[unseen_mask], all_preds[unseen_mask]) * 100
        print(f"\n  UNSEEN LANGUAGES ({len(unseen)} langs): {unseen_acc:.2f}%")
        unseen_labels = sorted(set(y[unseen_mask]))
        unseen_names = [all_langs[i] for i in unseen_labels]
        print(classification_report(y[unseen_mask], all_preds[unseen_mask],
              labels=unseen_labels, target_names=unseen_names, zero_division=0))
    
    # 7. Confusion Matrix
    print("Generating confusion matrix...")
    cm = confusion_matrix(y, all_preds, labels=list(range(len(all_langs))))
    cm_norm = cm.astype('float') / cm.sum(axis=1, keepdims=True)
    cm_norm = np.nan_to_num(cm_norm)
    
    fig, ax = plt.subplots(figsize=(max(10, len(all_langs)), max(8, len(all_langs) * 0.8)))
    sns.heatmap(cm_norm, annot=True, fmt='.2f', cmap='Blues',
                xticklabels=all_langs, yticklabels=all_langs, ax=ax, vmin=0, vmax=1)
    ax.set_xlabel('Predicted Language', fontsize=12)
    ax.set_ylabel('True Language', fontsize=12)
    ax.set_title(f'Gated Fusion Zero-Shot: {dataset_name}\n'
                 f'Overall: {overall_acc:.1f}% | '
                 f'Seen: {seen_acc:.1f}% | Unseen: {unseen_acc:.1f}%' 
                 if unseen_mask.sum() > 0 else
                 f'Gated Fusion Zero-Shot: {dataset_name} | Accuracy: {overall_acc:.1f}%',
                 fontsize=13)
    
    # Red line between seen and unseen
    if len(seen) > 0 and len(unseen) > 0:
        ax.axhline(y=len(seen), color='red', linewidth=2, linestyle='--')
        ax.axvline(x=len(seen), color='red', linewidth=2, linestyle='--')
    
    plt.tight_layout()
    out_png = f'{dataset_name}_confusion_matrix.png'
    plt.savefig(out_png, dpi=150)
    print(f"Saved confusion matrix to {out_png}")
    
    # Final summary
    print(f"\n{'='*60}")
    print(f"FINAL SUMMARY: {dataset_name}")
    print(f"{'='*60}")
    print(f"  Overall ({len(all_langs)} langs): {overall_acc:.2f}%")
    if seen_mask.sum() > 0:
        print(f"  Seen ({len(seen)} langs):    {seen_acc:.2f}%")
    if unseen_mask.sum() > 0:
        print(f"  Unseen ({len(unseen)} langs):  {unseen_acc:.2f}%")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True, 
                       help="Path to dataset folder (e.g., FLEURS_Indian, VoxLingua_Indian, CommonVoice_Indian, Vaani_Indian)")
    args = parser.parse_args()
    evaluate(args)
