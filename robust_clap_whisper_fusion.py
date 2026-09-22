"""
Robust Gated Fusion for Zero-Shot Language Identification
Implements techniques to solve cross-corpus domain shift (EkStep -> Vaani/YouTube).
"""
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import os
from msclap import CLAP
from sklearn.metrics import accuracy_score

SEEN_LANGS = ['Marathi', 'Kannada', 'English', 'Odia', 'Punjabi', 'Malayalam', 'Telugu', 'Gujarati']
DEVICE = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')

# Multi-Prompt Templates for Ensembling
PROMPT_TEMPLATES = [
    "A person speaking in {}",
    "Speech in the {} language",
    "Someone speaking {}",
    "A recording of {} speech",
    "Spoken {}",
    "A voice talking in {}"
]

class CLAPWhisperGatedFusion(nn.Module):
    def __init__(self, clap_dim=1024, whisper_dim=512, out_dim=1024, dropout=0.3):
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

def augment_features(features, noise_std=0.05, dropout_p=0.1):
    """Corrupts features to prevent memorization of dataset-specific acoustics."""
    noise = torch.randn_like(features) * noise_std
    features = features + noise
    features = F.dropout(features, p=dropout_p, training=True)
    return features

def get_ensembled_text_embeddings(clap_model, langs):
    """Averages multiple prompts per language for a robust text anchor."""
    print("Generating ensembled text embeddings...")
    all_embeddings = []
    for lang in langs:
        prompts = [template.format(lang) for template in PROMPT_TEMPLATES]
        with torch.no_grad():
            embs = clap_model.get_text_embeddings(prompts).detach()
            # Average the embeddings for this language
            avg_emb = embs.mean(dim=0)
            # Re-normalize
            avg_emb = F.normalize(avg_emb, p=2, dim=0)
            all_embeddings.append(avg_emb)
    
    return torch.stack(all_embeddings).to(DEVICE)

def contrastive_loss(audio_features, text_features, logit_scale, labels):
    audio_features = F.normalize(audio_features, p=2, dim=1)
    text_features = F.normalize(text_features, p=2, dim=1)
    logits = logit_scale.exp() * audio_features @ text_features.T
    loss = F.cross_entropy(logits, labels)
    return loss

def train_robust_fusion():
    print("="*60)
    print("TRAINING ROBUST GATED FUSION (Cross-Corpus Optimized)")
    print("="*60)
    
    clap_model = CLAP(version='2023', use_cuda=False)
    
    # 1. Multi-Prompt Ensembling
    text_embeddings = get_ensembled_text_embeddings(clap_model, SEEN_LANGS)
    
    print("\nLoading cached features...")
    train_data = np.load("features_fusion_ekstep.npz")
    X_c_train = train_data['X_clap']
    X_w_train = train_data['X_whisper']
    y_train = train_data['y']
    
    # Validation data (Optional, we'll use YouTube just to monitor)
    val_data = np.load("features_fusion_youtube.npz")
    X_c_val = val_data['X_clap']
    X_w_val = val_data['X_whisper']
    y_val = val_data['y']
    
    dataset_train = torch.utils.data.TensorDataset(
        torch.FloatTensor(X_c_train), torch.FloatTensor(X_w_train), torch.LongTensor(y_train)
    )
    loader_train = torch.utils.data.DataLoader(dataset_train, batch_size=256, shuffle=True)
    
    dataset_val = torch.utils.data.TensorDataset(
        torch.FloatTensor(X_c_val), torch.FloatTensor(X_w_val), torch.LongTensor(y_val)
    )
    loader_val = torch.utils.data.DataLoader(dataset_val, batch_size=256, shuffle=False)
    
    model = CLAPWhisperGatedFusion(clap_dim=1024, whisper_dim=512, out_dim=1024, dropout=0.4).to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-3) # Higher weight decay
    
    epochs = 40
    best_acc = 0.0
    
    print("\nStarting Training...")
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for c_batch, w_batch, labels_batch in loader_train:
            c_batch, w_batch, labels_batch = c_batch.to(DEVICE), w_batch.to(DEVICE), labels_batch.to(DEVICE)
            
            # FEATURE SPACE AUGMENTATION
            # Only augment CLAP features to force reliance on Whisper gate
            c_batch_aug = augment_features(c_batch, noise_std=0.08, dropout_p=0.2)
            
            optimizer.zero_grad()
            fused_audio = model(c_batch_aug, w_batch)
            
            loss = contrastive_loss(fused_audio, text_embeddings, model.logit_scale, labels_batch)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            
        model.eval()
        all_preds = []
        with torch.no_grad():
            for c_batch, w_batch, _ in loader_val:
                c_batch, w_batch = c_batch.to(DEVICE), w_batch.to(DEVICE)
                fused_audio = model(c_batch, w_batch)
                
                logits = model.logit_scale.exp() * fused_audio @ text_embeddings.T
                preds = torch.argmax(logits, dim=1)
                all_preds.extend(preds.cpu().numpy())
                
        val_acc = accuracy_score(y_val, all_preds) * 100
        print(f"Epoch {epoch+1:02d}/{epochs} | Loss: {total_loss/len(loader_train):.4f} | Zero-Shot Val Acc: {val_acc:.2f}%")
        
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), "robust_fusion_model_best.pt")
            
    print(f"\nTraining Complete! Best Val Accuracy: {best_acc:.2f}%")
    print("Saved robust weights to: robust_fusion_model_best.pt")

if __name__ == "__main__":
    train_robust_fusion()
