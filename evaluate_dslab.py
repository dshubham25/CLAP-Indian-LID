"""
DSLAB109 Evaluation Script
==========================
Runs the trained Gated Fusion model on Vaani and VoxLingua107 datasets
that are already stored on the dslab109 server.

Usage (on dslab109):
  python evaluate_dslab.py --dataset vaani
  python evaluate_dslab.py --dataset voxlingua
  python evaluate_dslab.py --dataset both
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
import matplotlib
matplotlib.use('Agg')  # No GUI on server
import matplotlib.pyplot as plt
import zipfile
import glob

# ============================================================
# DSLAB109 PATHS - EDIT THESE IF PATHS CHANGE
# ============================================================
VAANI_DIR = "/DATA/s23044/odyssey26/dataset/audio/vaani_unseen"
VOXLINGUA_ZIP_DIR = "/DATA/s23044/lid_40/zip"
VOXLINGUA_EXTRACT_DIR = os.path.expanduser("~/clap_fusion/voxlingua_extracted")

# ============================================================
# MODEL CONFIG
# ============================================================
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# The 8 languages the model was TRAINED on (from EkStep)
SEEN_LANGS = {'Marathi', 'Kannada', 'English', 'Odia', 'Punjabi', 'Malayalam', 'Telugu', 'Gujarati'}

# Mapping of common folder name patterns to standardized language names
FOLDER_NAME_MAP = {
    # Standard names
    "hindi": "Hindi", "bengali": "Bengali", "marathi": "Marathi",
    "telugu": "Telugu", "tamil": "Tamil", "gujarati": "Gujarati",
    "urdu": "Urdu", "kannada": "Kannada", "odia": "Odia",
    "malayalam": "Malayalam", "punjabi": "Punjabi", "assamese": "Assamese",
    "maithili": "Maithili", "santali": "Santali", "kashmiri": "Kashmiri",
    "nepali": "Nepali", "sindhi": "Sindhi", "konkani": "Konkani",
    "dogri": "Dogri", "manipuri": "Manipuri", "bodo": "Bodo",
    "sanskrit": "Sanskrit", "english": "English",
    # ISO codes
    "hi": "Hindi", "bn": "Bengali", "mr": "Marathi", "te": "Telugu",
    "ta": "Tamil", "gu": "Gujarati", "ur": "Urdu", "kn": "Kannada",
    "or": "Odia", "ori": "Odia", "odi": "Odia", "ml": "Malayalam",
    "pa": "Punjabi", "pun": "Punjabi", "as": "Assamese", "asm": "Assamese",
    "mai": "Maithili", "sat": "Santali", "ks": "Kashmiri", "kas": "Kashmiri",
    "ne": "Nepali", "sd": "Sindhi", "kok": "Konkani", "doi": "Dogri",
    "mni": "Manipuri", "brx": "Bodo", "sa": "Sanskrit", "san": "Sanskrit",
    "en": "English", "eng": "English",
    "hin": "Hindi", "ben": "Bengali", "mar": "Marathi", "tel": "Telugu",
    "tam": "Tamil", "guj": "Gujarati", "kan": "Kannada", "mal": "Malayalam",
}

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
    """Auto-discover languages from folder structure."""
    # First pass: discover all folders and map to standardized names
    lang_to_dirs = {}  # lang_name -> list of (dir_path, num_files)
    
    for d in sorted(os.listdir(dataset_dir)):
        full_path = os.path.join(dataset_dir, d)
        if os.path.isdir(full_path) and not d.startswith('.'):
            # Find audio files recursively
            audio_files = []
            for ext in ['*.wav', '*.flac', '*.mp3', '*.ogg']:
                audio_files.extend(glob.glob(os.path.join(full_path, '**', ext), recursive=True))
            
            if len(audio_files) == 0:
                continue
            
            # Map folder name to standardized language name
            folder_lower = d.lower().strip()
            
            # Strip numeric suffixes like _2, _3, _5 (common in Vaani state splits)
            import re
            base_name = re.sub(r'_\d+$', '', folder_lower)
            
            lang_name = FOLDER_NAME_MAP.get(folder_lower, 
                        FOLDER_NAME_MAP.get(base_name, base_name.title()))
            
            if lang_name not in lang_to_dirs:
                lang_to_dirs[lang_name] = []
            lang_to_dirs[lang_name].append((full_path, len(audio_files)))
            print(f"  Found: {d}/ ({len(audio_files)} files) -> {lang_name}")
    
    # Second pass: build final list with merged directories
    langs = []
    for lang_name in sorted(lang_to_dirs.keys()):
        dirs = lang_to_dirs[lang_name]
        total_files = sum(n for _, n in dirs)
        all_dirs = [d for d, _ in dirs]
        langs.append((lang_name, all_dirs, total_files))
    
    print(f"\n  Total: {len(langs)} unique languages discovered")
    for lang_name, dirs, n in langs:
        status = "SEEN" if lang_name in SEEN_LANGS else "UNSEEN"
        print(f"    [{status}] {lang_name}: {n} files ({len(dirs)} folder(s))")
    
    return langs

def extract_features(lang_info_list, cache_name, max_per_lang=500):
    """Extract CLAP + Whisper features."""
    if os.path.exists(cache_name):
        print(f"\nLoading cached features from {cache_name}...")
        data = np.load(cache_name, allow_pickle=True)
        return data['X_clap'], data['X_whisper'], data['y'], list(data['all_langs']), data['lang_per_sample']
    
    print(f"\nInitializing models for feature extraction...")
    
    clap_model = CLAP(version='2023', use_cuda=(DEVICE.type == 'cuda'))
    clap_model.read_audio = types.MethodType(custom_read_audio, clap_model)
    
    whisper_id = "openai/whisper-base"
    processor = AutoFeatureExtractor.from_pretrained(whisper_id)
    whisper_model = WhisperModel.from_pretrained(whisper_id).encoder.to(DEVICE)
    whisper_model.eval()
    
    all_langs = [info[0] for info in lang_info_list]
    lang_to_idx = {l: i for i, l in enumerate(all_langs)}
    
    # Gather files (each lang may have multiple directories)
    files_to_process = []
    for lang_name, lang_dirs, _ in lang_info_list:
        audio_files = []
        for lang_dir in lang_dirs:
            for ext in ['*.wav', '*.flac', '*.mp3', '*.ogg']:
                audio_files.extend(glob.glob(os.path.join(lang_dir, '**', ext), recursive=True))
        audio_files.sort()
        
        # Cap at max_per_lang
        for f in audio_files[:max_per_lang]:
            files_to_process.append((f, lang_to_idx[lang_name], lang_name))
    
    print(f"Processing {len(files_to_process)} files across {len(all_langs)} languages...")
    
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
    lang_per_sample = np.array(lang_names_list)
    
    np.savez(cache_name, X_clap=X_clap, X_whisper=X_whisper, y=y, 
             all_langs=np.array(all_langs), lang_per_sample=lang_per_sample)
    print(f"Saved features to {cache_name}")
    return X_clap, X_whisper, y, all_langs, lang_per_sample

def prepare_voxlingua():
    """Extract VoxLingua107 zips if needed."""
    if not os.path.exists(VOXLINGUA_ZIP_DIR):
        print(f"ERROR: VoxLingua zip dir not found: {VOXLINGUA_ZIP_DIR}")
        return None
    
    os.makedirs(VOXLINGUA_EXTRACT_DIR, exist_ok=True)
    
    zip_files = sorted(glob.glob(os.path.join(VOXLINGUA_ZIP_DIR, "*.zip")))
    if not zip_files:
        # Maybe they're already extracted directories
        print(f"No zip files found. Checking if already extracted...")
        if os.path.isdir(VOXLINGUA_ZIP_DIR):
            return VOXLINGUA_ZIP_DIR
        return None
    
    print(f"Found {len(zip_files)} zip files. Extracting Indian languages...")
    
    for zf_path in zip_files:
        lang_code = os.path.splitext(os.path.basename(zf_path))[0].lower()
        if lang_code in FOLDER_NAME_MAP:
            lang_name = FOLDER_NAME_MAP[lang_code]
            extract_to = os.path.join(VOXLINGUA_EXTRACT_DIR, lang_name.lower())
            
            if os.path.exists(extract_to) and len(os.listdir(extract_to)) > 0:
                print(f"  {lang_name}: Already extracted, skipping...")
                continue
            
            print(f"  Extracting {lang_name} from {os.path.basename(zf_path)}...")
            os.makedirs(extract_to, exist_ok=True)
            try:
                with zipfile.ZipFile(zf_path, 'r') as zf:
                    zf.extractall(extract_to)
            except Exception as e:
                print(f"    ERROR: {e}")
    
    return VOXLINGUA_EXTRACT_DIR

def evaluate(dataset_dir, dataset_name, model_path="fusion_model_best.pt"):
    print("="*60)
    print(f"EVALUATING GATED FUSION ON: {dataset_name}")
    print(f"Dataset path: {dataset_dir}")
    print(f"Device: {DEVICE}")
    print("="*60)
    
    # 1. Discover languages
    print(f"\nDiscovering languages in {dataset_dir}/...")
    lang_info = discover_languages(dataset_dir)
    
    if len(lang_info) == 0:
        print(f"ERROR: No audio folders found in {dataset_dir}/")
        return
    
    all_langs = [info[0] for info in lang_info]
    seen = [l for l in all_langs if l in SEEN_LANGS]
    unseen = [l for l in all_langs if l not in SEEN_LANGS]
    print(f"\n  SEEN languages ({len(seen)}):   {seen}")
    print(f"  UNSEEN languages ({len(unseen)}): {unseen}")
    
    # 2. Text embeddings for ALL discovered languages (Robust Multi-Prompt & Centering)
    print(f"\nGenerating ensembled text embeddings for {len(all_langs)} languages...")
    clap_model = CLAP(version='2023', use_cuda=(DEVICE.type == 'cuda'))
    
    PROMPT_TEMPLATES = [
        "A person speaking in {}", "Speech in the {} language", "Someone speaking {}",
        "A recording of {} speech", "Spoken {}", "A voice talking in {}"
    ]
    
    all_text_embs = []
    for lang in all_langs:
        prompts = [template.format(lang) for template in PROMPT_TEMPLATES]
        with torch.no_grad():
            embs = clap_model.get_text_embeddings(prompts).detach().to(DEVICE)
            avg_emb = embs.mean(dim=0)
            avg_emb = F.normalize(avg_emb, p=2, dim=0)
            all_text_embs.append(avg_emb)
    
    text_embeddings = torch.stack(all_text_embs)
    
    del clap_model
    
    # 3. Extract features
    cache_name = f"features_fusion_{dataset_name}.npz"
    X_clap, X_whisper, y, all_langs, lang_per_sample = extract_features(
        lang_info, cache_name, max_per_lang=500
    )
    
    # 4. Load trained model
    print(f"\nLoading trained model from {model_path}...")
    model = CLAPWhisperGatedFusion(clap_dim=1024, whisper_dim=512, out_dim=1024).to(DEVICE)
    model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    model.eval()
    
    # 5. Inference
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
    
    # 6. Split results
    seen_mask = np.array([lang_per_sample[i] in SEEN_LANGS for i in range(len(lang_per_sample))])
    unseen_mask = ~seen_mask
    
    overall_acc = accuracy_score(y, all_preds) * 100
    
    print(f"{'='*60}")
    print(f"RESULTS: {dataset_name}")
    print(f"{'='*60}")
    print(f"\n  OVERALL ACCURACY ({len(all_langs)} languages): {overall_acc:.2f}%")
    
    seen_acc, unseen_acc = 0, 0
    
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
    
    fig_size = max(10, len(all_langs) * 0.7)
    fig, ax = plt.subplots(figsize=(fig_size, fig_size * 0.85))
    sns.heatmap(cm_norm, annot=True, fmt='.2f', cmap='Blues',
                xticklabels=all_langs, yticklabels=all_langs, ax=ax, vmin=0, vmax=1)
    ax.set_xlabel('Predicted Language', fontsize=12)
    ax.set_ylabel('True Language', fontsize=12)
    
    title = f'Gated Fusion Zero-Shot: {dataset_name} | Overall: {overall_acc:.1f}%'
    if seen_mask.sum() > 0 and unseen_mask.sum() > 0:
        title += f'\nSeen: {seen_acc:.1f}% | Unseen: {unseen_acc:.1f}%'
    ax.set_title(title, fontsize=13)
    
    if len(seen) > 0 and len(unseen) > 0:
        ax.axhline(y=len(seen), color='red', linewidth=2, linestyle='--')
        ax.axvline(x=len(seen), color='red', linewidth=2, linestyle='--')
    
    plt.tight_layout()
    out_png = f'{dataset_name}_confusion_matrix.png'
    plt.savefig(out_png, dpi=150)
    print(f"Saved confusion matrix to {out_png}")
    
    # Summary
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
    parser.add_argument("--dataset", required=True, choices=["vaani", "voxlingua", "common_voice", "all"],
                       help="Which dataset to evaluate on")
    parser.add_argument("--model", default="fusion_model_best.pt",
                       help="Path to trained model weights")
    args = parser.parse_args()
    
    if args.dataset in ["vaani", "all"]:
        if os.path.exists(VAANI_DIR):
            evaluate(VAANI_DIR, "Vaani_Unseen", args.model)
        else:
            print(f"ERROR: Vaani directory not found at {VAANI_DIR}")
    
    if args.dataset in ["voxlingua", "all"]:
        vox_dir = prepare_voxlingua()
        if vox_dir:
            evaluate(vox_dir, "VoxLingua107", args.model)
        else:
            print(f"ERROR: Could not find/extract VoxLingua107 data")
            
    if args.dataset in ["common_voice", "all"]:
        cv_dir = os.path.expanduser("~/clap_fusion/common_voice_extracted")
        if os.path.exists(cv_dir):
            evaluate(cv_dir, "CommonVoice", args.model)
        else:
            print(f"ERROR: Common Voice dir not found. Run download_cv.py first!")
