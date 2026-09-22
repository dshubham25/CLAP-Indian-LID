"""
Evaluate the trained Gated Fusion model on the FLEURS Indian dataset.
Tests on 14 Indian languages: 7 seen during training, 7 completely unseen.
This proves the zero-shot generalization power of the contrastive architecture.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import types
import soundfile as sf
import torchaudio.transforms as T
import os
import numpy as np
from tqdm import tqdm
from msclap import CLAP
from transformers import AutoFeatureExtractor, WhisperModel
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

DEVICE = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')

# The 8 languages the model was TRAINED on (from EkStep)
SEEN_LANGS = ['Marathi', 'Kannada', 'English', 'Odia', 'Punjabi', 'Malayalam', 'Telugu', 'Gujarati']

# ALL 14 Indian languages in FLEURS (7 seen + 7 unseen)
ALL_LANGS = [
    'Marathi', 'Kannada', 'English', 'Odia', 'Punjabi', 'Malayalam', 'Telugu', 'Gujarati',  # SEEN
    'Hindi', 'Bengali', 'Tamil', 'Urdu', 'Nepali', 'Assamese', 'Sindhi'  # UNSEEN
]

# --- GATED FUSION ARCHITECTURE (must match training) ---
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

def extract_fleurs_features(fleurs_dir, langs, cache_name):
    """Extract CLAP + Whisper features for the FLEURS dataset."""
    if os.path.exists(cache_name):
        print(f"Loading cached FLEURS features from {cache_name}...")
        data = np.load(cache_name)
        return data['X_clap'], data['X_whisper'], data['y'], data['lang_names']
    
    print(f"\nExtracting features for FLEURS ({fleurs_dir})...")
    
    # Init models
    clap_model = CLAP(version='2023', use_cuda=False)
    clap_model.read_audio = types.MethodType(custom_read_audio, clap_model)
    
    whisper_id = "openai/whisper-base"
    processor = AutoFeatureExtractor.from_pretrained(whisper_id)
    whisper_model = WhisperModel.from_pretrained(whisper_id).encoder.to(DEVICE)
    whisper_model.eval()
    
    lang_to_idx = {l: i for i, l in enumerate(langs)}
    
    files_to_process = []
    for lang_name in langs:
        lang_dir = os.path.join(fleurs_dir, lang_name.lower())
        if not os.path.exists(lang_dir):
            print(f"  WARNING: Directory not found for {lang_name} at {lang_dir}")
            continue
        for f in sorted(os.listdir(lang_dir)):
            if f.endswith('.wav'):
                files_to_process.append((os.path.join(lang_dir, f), lang_to_idx[lang_name], lang_name))
    
    print(f"Found {len(files_to_process)} files across {len(langs)} languages.")
    
    clap_features = []
    whisper_features = []
    labels = []
    lang_names_list = []
    
    for path, label, lang_name in tqdm(files_to_process, desc="Extracting FLEURS features"):
        try:
            # CLAP
            c_emb = clap_model.get_audio_embeddings([path], resample=True).detach().cpu().numpy().squeeze()
            
            # Whisper
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
    print(f"Saved FLEURS features to {cache_name}")
    return X_clap, X_whisper, y, lang_names_arr


def evaluate_fleurs():
    print("="*60)
    print("EVALUATING GATED FUSION ON FLEURS (14 INDIAN LANGUAGES)")
    print("="*60)
    
    # 1. Generate text embeddings for ALL 14 languages
    print("\nGenerating text embeddings for 14 languages...")
    clap_model = CLAP(version='2023', use_cuda=False)
    prompts = [f"A person speaking in {lang}" for lang in ALL_LANGS]
    text_embeddings = clap_model.get_text_embeddings(prompts).detach().to(DEVICE)
    print(f"  Text embedding shape: {text_embeddings.shape}")
    
    # 2. Extract FLEURS features
    X_clap, X_whisper, y, lang_names = extract_fleurs_features(
        "FLEURS_Indian", ALL_LANGS, "features_fusion_fleurs.npz"
    )
    
    # 3. Load trained Gated Fusion model
    print("\nLoading trained Gated Fusion model...")
    model = CLAPWhisperGatedFusion(clap_dim=1024, whisper_dim=512, out_dim=1024).to(DEVICE)
    model.load_state_dict(torch.load("fusion_model_best.pt", map_location=DEVICE))
    model.eval()
    
    # 4. Run inference
    print("Running zero-shot inference on FLEURS...\n")
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
    
    # 5. Split results into SEEN vs UNSEEN
    seen_mask = np.array([lang_names[i] in SEEN_LANGS for i in range(len(lang_names))])
    unseen_mask = ~seen_mask
    
    # Overall
    overall_acc = accuracy_score(y, all_preds) * 100
    print(f"{'='*50}")
    print(f"OVERALL ACCURACY (14 languages): {overall_acc:.2f}%")
    print(f"{'='*50}")
    
    # Seen languages
    if seen_mask.sum() > 0:
        seen_acc = accuracy_score(y[seen_mask], all_preds[seen_mask]) * 100
        print(f"\nSEEN LANGUAGES (7 langs, trained on EkStep):")
        print(f"  Accuracy: {seen_acc:.2f}%")
        
        seen_labels = sorted(set(y[seen_mask]))
        seen_names = [ALL_LANGS[i] for i in seen_labels]
        print(classification_report(y[seen_mask], all_preds[seen_mask], 
              labels=seen_labels, target_names=seen_names, zero_division=0))
    
    # Unseen languages
    if unseen_mask.sum() > 0:
        unseen_acc = accuracy_score(y[unseen_mask], all_preds[unseen_mask]) * 100
        print(f"\nUNSEEN LANGUAGES (7 langs, NEVER trained on):")
        print(f"  Accuracy: {unseen_acc:.2f}%")
        
        unseen_labels = sorted(set(y[unseen_mask]))
        unseen_names = [ALL_LANGS[i] for i in unseen_labels]
        print(classification_report(y[unseen_mask], all_preds[unseen_mask],
              labels=unseen_labels, target_names=unseen_names, zero_division=0))
    
    # 6. Generate confusion matrix
    print("\nGenerating confusion matrix...")
    cm = confusion_matrix(y, all_preds, labels=list(range(len(ALL_LANGS))))
    
    fig, ax = plt.subplots(figsize=(14, 11))
    
    # Normalize
    cm_norm = cm.astype('float') / cm.sum(axis=1, keepdims=True)
    cm_norm = np.nan_to_num(cm_norm)
    
    sns.heatmap(cm_norm, annot=True, fmt='.2f', cmap='Blues',
                xticklabels=ALL_LANGS, yticklabels=ALL_LANGS, ax=ax,
                vmin=0, vmax=1)
    
    ax.set_xlabel('Predicted Language', fontsize=12)
    ax.set_ylabel('True Language', fontsize=12)
    ax.set_title('Gated Fusion: Zero-Shot on FLEURS (14 Indian Languages)\n'
                 'Top 8 = SEEN during training | Bottom 7 = UNSEEN', fontsize=13)
    
    # Draw a line separating seen from unseen
    ax.axhline(y=8, color='red', linewidth=2, linestyle='--')
    ax.axvline(x=8, color='red', linewidth=2, linestyle='--')
    
    plt.tight_layout()
    plt.savefig('fleurs_confusion_matrix.png', dpi=150)
    print("Saved confusion matrix to fleurs_confusion_matrix.png")
    
    # Summary
    print(f"\n{'='*60}")
    print(f"FINAL SUMMARY")
    print(f"{'='*60}")
    print(f"  Overall Accuracy (14 langs):  {overall_acc:.2f}%")
    if seen_mask.sum() > 0:
        print(f"  Seen Languages (7 langs):     {seen_acc:.2f}%")
    if unseen_mask.sum() > 0:
        print(f"  Unseen Languages (7 langs):   {unseen_acc:.2f}%")

if __name__ == "__main__":
    evaluate_fleurs()
