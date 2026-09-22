"""
Train on 80% EkStep, Evaluate on 20% EkStep Holdout.
"""
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from msclap import CLAP
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split

SEEN_LANGS = ['Marathi', 'Kannada', 'English', 'Odia', 'Punjabi', 'Malayalam', 'Telugu', 'Gujarati']
DEVICE = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')

PROMPT_TEMPLATES = [
    "A person speaking in {}",
    "Speech in the {} language",
    "Someone speaking {}",
    "A recording of {} speech",
    "Spoken {}",
    "A voice talking in {}"
]

class CLAPWhisperGatedFusion(nn.Module):
    def __init__(self, clap_dim=1024, whisper_dim=512, out_dim=1024, dropout=0.4):
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

def augment_features(features, noise_std=0.08, dropout_p=0.2):
    noise = torch.randn_like(features) * noise_std
    features = features + noise
    features = F.dropout(features, p=dropout_p, training=True)
    return features

def get_ensembled_text_embeddings(clap_model, langs):
    print("Generating ensembled text embeddings...")
    all_embeddings = []
    for lang in langs:
        prompts = [template.format(lang) for template in PROMPT_TEMPLATES]
        with torch.no_grad():
            embs = clap_model.get_text_embeddings(prompts).detach()
            avg_emb = embs.mean(dim=0)
            avg_emb = F.normalize(avg_emb, p=2, dim=0)
            all_embeddings.append(avg_emb)
    return torch.stack(all_embeddings).to(DEVICE)

def contrastive_loss(audio_features, text_features, logit_scale, labels):
    audio_features = F.normalize(audio_features, p=2, dim=1)
    text_features = F.normalize(text_features, p=2, dim=1)
    logits = logit_scale.exp() * audio_features @ text_features.T
    return F.cross_entropy(logits, labels)

def main():
    print("="*60)
    print("EKSTEP 80/20 SPLIT EVALUATION")
    print("="*60)
    
    clap_model = CLAP(version='2023', use_cuda=False)
    text_embeddings = get_ensembled_text_embeddings(clap_model, SEEN_LANGS)
    
    print("\nLoading EkStep cached features...")
    data = np.load("features_fusion_ekstep.npz")
    X_c, X_w, y = data['X_clap'], data['X_whisper'], data['y']
    
    # SPLIT 80/20
    X_c_train, X_c_test, X_w_train, X_w_test, y_train, y_test = train_test_split(
        X_c, X_w, y, test_size=0.20, random_state=42, stratify=y
    )
    
    print(f"Training on {len(y_train)} samples, Testing on {len(y_test)} samples...")
    
    dataset_train = torch.utils.data.TensorDataset(
        torch.FloatTensor(X_c_train), torch.FloatTensor(X_w_train), torch.LongTensor(y_train)
    )
    loader_train = torch.utils.data.DataLoader(dataset_train, batch_size=256, shuffle=True)
    
    dataset_test = torch.utils.data.TensorDataset(
        torch.FloatTensor(X_c_test), torch.FloatTensor(X_w_test), torch.LongTensor(y_test)
    )
    loader_test = torch.utils.data.DataLoader(dataset_test, batch_size=256, shuffle=False)
    
    model = CLAPWhisperGatedFusion().to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-3)
    
    epochs = 25  # Should be enough for in-domain
    best_acc = 0.0
    
    print("\nStarting Training...")
    for epoch in range(epochs):
        model.train()
        for c_batch, w_batch, labels_batch in loader_train:
            c_batch, w_batch, labels_batch = c_batch.to(DEVICE), w_batch.to(DEVICE), labels_batch.to(DEVICE)
            c_batch_aug = augment_features(c_batch)
            
            optimizer.zero_grad()
            fused_audio = model(c_batch_aug, w_batch)
            loss = contrastive_loss(fused_audio, text_embeddings, model.logit_scale, labels_batch)
            loss.backward()
            optimizer.step()
            
        # Eval on 20% holdout
        model.eval()
        all_preds = []
        with torch.no_grad():
            for c_batch, w_batch, _ in loader_test:
                c_batch, w_batch = c_batch.to(DEVICE), w_batch.to(DEVICE)
                fused_audio = model(c_batch, w_batch)
                logits = model.logit_scale.exp() * fused_audio @ text_embeddings.T
                preds = torch.argmax(logits, dim=1)
                all_preds.extend(preds.cpu().numpy())
                
        val_acc = accuracy_score(y_test, all_preds) * 100
        print(f"Epoch {epoch+1:02d}/{epochs} | EkStep 20% Holdout Acc: {val_acc:.2f}%")
        
        if val_acc > best_acc:
            best_acc = val_acc
            best_preds = all_preds
            
    print(f"\n{'='*60}")
    print(f"FINAL IN-DOMAIN RESULTS (EKSTEP 20% HOLDOUT)")
    print(f"{'='*60}")
    print(f"Overall Accuracy: {best_acc:.2f}%\n")
    print(classification_report(y_test, best_preds, target_names=SEEN_LANGS, digits=4))

if __name__ == "__main__":
    main()
