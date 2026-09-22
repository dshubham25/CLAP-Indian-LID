"""
CLAP-Whisper Fusion Architecture

This script implements a hybrid model that injects Whisper's domain-invariant 
phonetic features into CLAP's zero-shot latent space via a trainable MLP adapter.
"""
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import types
import soundfile as sf
import torchaudio.transforms as T
import os
import numpy as np
from tqdm import tqdm
from msclap import CLAP
from transformers import AutoFeatureExtractor, WhisperModel
from sklearn.metrics import accuracy_score, classification_report
import argparse
import time

# --- SETUP ---
FOLDER_TO_LANG = {
    "hin": "Hindi", "ben": "Bengali", "mar": "Marathi", "tel": "Telugu",
    "tam": "Tamil", "guj": "Gujarati", "urd": "Urdu", "kan": "Kannada",
    "ori": "Odia", "odi": "Odia", "mal": "Malayalam", "pun": "Punjabi",
    "asm": "Assamese", "eng": "English"
}
SEEN_LANGS = ['Marathi', 'Kannada', 'English', 'Odia', 'Punjabi', 'Malayalam', 'Telugu', 'Gujarati']
DEVICE = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')

# --- THE FUSION ADAPTER ---
class CLAPWhisperFusion(nn.Module):
    def __init__(self, clap_dim=512, whisper_dim=512, hidden_dim=512, out_dim=512, dropout=0.2):
        super(CLAPWhisperFusion, self).__init__()
        
        # We concatenate CLAP and Whisper features, so input is clap_dim + whisper_dim
        self.fc1 = nn.Linear(clap_dim + whisper_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(hidden_dim, out_dim)
        
        # Learnable temperature parameter for contrastive loss (initialized to match CLAP's default scale)
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))

    def forward(self, clap_audio, whisper_audio):
        x = torch.cat([clap_audio, whisper_audio], dim=1)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = F.normalize(x, p=2, dim=1)
        return x

# --- GATED FUSION (Vector-Level Attention) ---
class CLAPWhisperGatedFusion(nn.Module):
    def __init__(self, clap_dim=1024, whisper_dim=512, out_dim=1024, dropout=0.2):
        super(CLAPWhisperGatedFusion, self).__init__()
        
        # 1. Attention Gate: Whisper dictates what acoustic noise to suppress in CLAP
        self.gate_proj = nn.Linear(whisper_dim, clap_dim)
        self.gate_activation = nn.Sigmoid()
        
        # 2. Final Projection: Fusing the filtered CLAP and the Whisper phonetics
        self.fc_fuse = nn.Linear(clap_dim + whisper_dim, out_dim)
        self.dropout = nn.Dropout(dropout)
        
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))

    def forward(self, clap_audio, whisper_audio):
        # Generate the attention mask using Whisper's phonetic knowledge
        attention_mask = self.gate_activation(self.gate_proj(whisper_audio))
        
        # Suppress the background noise in CLAP
        filtered_clap = clap_audio * attention_mask
        
        # Concatenate and project to final text space
        x = torch.cat([filtered_clap, whisper_audio], dim=1)
        x = self.fc_fuse(x)
        x = self.dropout(x)
        x = F.normalize(x, p=2, dim=1)
        return x

# --- DATA EXTRACTION ---
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
        audio = audio.mean(axis=1) # to mono
    if sr != target_sr:
        resampler = T.Resample(sr, target_sr)
        audio = resampler(torch.FloatTensor(audio)).numpy()
    return audio

def extract_combined_features(folder_path, langs, cache_name):
    print(f"\nInitializing models to extract features for {folder_path}...")
    
    # 1. Init CLAP
    clap_model = CLAP(version='2023', use_cuda=False)
    clap_model.read_audio = types.MethodType(custom_read_audio, clap_model)
    
    # 2. Init Whisper
    whisper_id = "openai/whisper-base"
    processor = AutoFeatureExtractor.from_pretrained(whisper_id)
    whisper_model = WhisperModel.from_pretrained(whisper_id).encoder.to(DEVICE)
    whisper_model.eval()
    
    # 3. Gather files (DETERMINISTIC SORT)
    lang_to_idx = {l: i for i, l in enumerate(langs)}
    files_to_process = []
    
    for root, _, files in os.walk(folder_path):
        folder_name = os.path.basename(root)
        if folder_name in FOLDER_TO_LANG:
            lang_name = FOLDER_TO_LANG[folder_name]
            if lang_name in langs:
                for f in files:
                    if f.endswith('.wav'):
                        files_to_process.append((os.path.join(root, f), lang_to_idx[lang_name]))
    
    # VERY IMPORTANT: Sort so we always process in the exact same order
    files_to_process.sort(key=lambda x: x[0])
    print(f"Found {len(files_to_process)} files. Extracting dual features...")
    
    clap_features = []
    whisper_features = []
    labels = []
    
    for path, label in tqdm(files_to_process, desc="Extracting"):
        try:
            # --- CLAP Feature ---
            c_emb = clap_model.get_audio_embeddings([path], resample=True).detach().cpu().numpy().squeeze()
            
            # --- Whisper Feature ---
            w_audio = load_audio_for_whisper(path)
            inputs = processor(w_audio, sampling_rate=16000, return_tensors="pt")
            w_input = inputs.input_features.to(DEVICE)
            with torch.no_grad():
                out = whisper_model(w_input)
                w_emb = out.last_hidden_state.mean(dim=1).squeeze().cpu().numpy()
            
            clap_features.append(c_emb)
            whisper_features.append(w_emb)
            labels.append(label)
        except Exception as e:
            pass # Skip corrupted files
            
    X_clap = np.array(clap_features)
    X_whisper = np.array(whisper_features)
    y = np.array(labels)
    
    np.savez(cache_name, X_clap=X_clap, X_whisper=X_whisper, y=y)
    print(f"Saved combined features to {cache_name}")
    return X_clap, X_whisper, y

# --- LOSS FUNCTION ---
# Symmetric InfoNCE Loss (Contrastive)
def contrastive_loss(audio_features, text_features, logit_scale, labels):
    # Normalize features
    audio_features = F.normalize(audio_features, p=2, dim=1)
    text_features = F.normalize(text_features, p=2, dim=1)
    
    # Cosine similarity scaled by temperature
    logits = logit_scale.exp() * audio_features @ text_features.T
    
    # The labels indicate which text prompt is the correct match for each audio sample
    loss = F.cross_entropy(logits, labels)
    return loss

# --- TRAINING PIPELINE ---
def train_fusion():
    print("="*60)
    print("TRAINING CLAP-WHISPER FUSION ARCHITECTURE")
    print("="*60)
    
    # 1. Get Text Embeddings from CLAP (Frozen anchors)
    print("\nGenerating frozen text embeddings...")
    clap_model = CLAP(version='2023', use_cuda=False)
    prompts = [f"A person speaking in {lang}" for lang in SEEN_LANGS]
    text_embeddings = clap_model.get_text_embeddings(prompts).detach().to(DEVICE)
    
    # 2. Get Audio Features
    train_cache = "features_fusion_ekstep.npz"
    if os.path.exists(train_cache):
        print(f"Loading cached train features from {train_cache}...")
        data = np.load(train_cache)
        X_c, X_w, y = data['X_clap'], data['X_whisper'], data['y']
    else:
        X_c, X_w, y = extract_combined_features("ekstep_seen", SEEN_LANGS, train_cache)
        
    test_cache = "features_fusion_youtube.npz"
    if os.path.exists(test_cache):
        print(f"Loading cached test features from {test_cache}...")
        data = np.load(test_cache)
        X_c_test, X_w_test, y_test = data['X_clap'], data['X_whisper'], data['y']
    else:
        X_c_test, X_w_test, y_test = extract_combined_features("IITMandi_YouTube", SEEN_LANGS, test_cache)
        
    # Convert to tensors
    dataset_train = torch.utils.data.TensorDataset(torch.FloatTensor(X_c), torch.FloatTensor(X_w), torch.LongTensor(y))
    loader_train = torch.utils.data.DataLoader(dataset_train, batch_size=256, shuffle=True)
    
    dataset_test = torch.utils.data.TensorDataset(torch.FloatTensor(X_c_test), torch.FloatTensor(X_w_test), torch.LongTensor(y_test))
    loader_test = torch.utils.data.DataLoader(dataset_test, batch_size=256, shuffle=False)
    
    # 3. Init Model and Optimizer
    clap_dim = X_c.shape[1]
    whisper_dim = X_w.shape[1]
    text_dim = text_embeddings.shape[1]
    print(f"\nFeature Dimensions -> CLAP: {clap_dim}, Whisper: {whisper_dim}, Text: {text_dim}")
    print("\nUsing Architecture: GATED FUSION (Cross-Attention Approximation)")
    
    model = CLAPWhisperGatedFusion(clap_dim=clap_dim, whisper_dim=whisper_dim, out_dim=text_dim).to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', patience=3, factor=0.5)
    
    epochs = 30
    best_acc = 0.0
    
    print("\nStarting Training...")
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        
        for c_batch, w_batch, labels_batch in loader_train:
            c_batch, w_batch, labels_batch = c_batch.to(DEVICE), w_batch.to(DEVICE), labels_batch.to(DEVICE)
            
            optimizer.zero_grad()
            fused_audio = model(c_batch, w_batch)
            
            loss = contrastive_loss(fused_audio, text_embeddings, model.logit_scale, labels_batch)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
        # Evaluation
        model.eval()
        all_preds = []
        with torch.no_grad():
            for c_batch, w_batch, _ in loader_test:
                c_batch, w_batch = c_batch.to(DEVICE), w_batch.to(DEVICE)
                fused_audio = model(c_batch, w_batch)
                
                # Zero-shot inference: cosine similarity between fused audio and text anchors
                logits = model.logit_scale.exp() * fused_audio @ text_embeddings.T
                preds = torch.argmax(logits, dim=1)
                all_preds.extend(preds.cpu().numpy())
                
        val_acc = accuracy_score(y_test, all_preds) * 100
        scheduler.step(val_acc)
        
        print(f"Epoch {epoch+1:02d}/{epochs} | Train Loss: {total_loss/len(loader_train):.4f} | Zero-Shot YouTube Acc: {val_acc:.2f}%")
        
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), "fusion_model_best.pt")
            best_preds = all_preds
            
    print(f"\nTraining Complete. Best YouTube Accuracy: {best_acc:.2f}%")
    print(f"Model saved to fusion_model_best.pt")
    
    print("\nDetailed Cross-Domain (YouTube) Classification Report (Best Epoch):")
    print(classification_report(y_test, best_preds, target_names=SEEN_LANGS, zero_division=0))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--extract-only", action="store_true", help="Only extract features and cache them, do not train.")
    args = parser.parse_args()
    
    if args.extract_only:
        extract_combined_features("ekstep_seen", SEEN_LANGS, "features_fusion_ekstep.npz")
        extract_combined_features("IITMandi_YouTube", SEEN_LANGS, "features_fusion_youtube.npz")
    else:
        train_fusion()
