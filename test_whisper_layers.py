"""
Deep Dive #2: Layer-Wise Analysis of Whisper's Attention Gating
This script tests Whisper Layer 2 (early), Layer 4 (middle), and Layer 6 (late) 
as the attention gate for the CLAP acoustic features.
"""
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from msclap import CLAP
from sklearn.metrics import accuracy_score
import os

SEEN_LANGS = ['Marathi', 'Kannada', 'English', 'Odia', 'Punjabi', 'Malayalam', 'Telugu', 'Gujarati']
DEVICE = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')

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

def contrastive_loss(audio_features, text_features, logit_scale, labels):
    audio_features = F.normalize(audio_features, p=2, dim=1)
    text_features = F.normalize(text_features, p=2, dim=1)
    logits = logit_scale.exp() * audio_features @ text_features.T
    loss = F.cross_entropy(logits, labels)
    return loss

def evaluate_layer(layer_name, X_clap_train, X_w_train, y_train, X_clap_test, X_w_test, y_test, text_embeddings):
    print(f"\n{'='*50}")
    print(f"TRAINING GATED FUSION WITH WHISPER {layer_name.upper()}")
    print(f"{'='*50}")
    
    dataset_train = torch.utils.data.TensorDataset(torch.FloatTensor(X_clap_train), torch.FloatTensor(X_w_train), torch.LongTensor(y_train))
    loader_train = torch.utils.data.DataLoader(dataset_train, batch_size=256, shuffle=True)
    
    dataset_test = torch.utils.data.TensorDataset(torch.FloatTensor(X_clap_test), torch.FloatTensor(X_w_test), torch.LongTensor(y_test))
    loader_test = torch.utils.data.DataLoader(dataset_test, batch_size=256, shuffle=False)
    
    model = CLAPWhisperGatedFusion(clap_dim=1024, whisper_dim=512, out_dim=1024).to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', patience=3, factor=0.5)
    
    best_acc = 0.0
    epochs = 30
    
    for epoch in range(epochs):
        model.train()
        for c_batch, w_batch, labels_batch in loader_train:
            c_batch, w_batch, labels_batch = c_batch.to(DEVICE), w_batch.to(DEVICE), labels_batch.to(DEVICE)
            optimizer.zero_grad()
            fused_audio = model(c_batch, w_batch)
            loss = contrastive_loss(fused_audio, text_embeddings, model.logit_scale, labels_batch)
            loss.backward()
            optimizer.step()
            
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
        scheduler.step(val_acc)
        if val_acc > best_acc:
            best_acc = val_acc
            
    print(f"[{layer_name.upper()}] Best Zero-Shot YouTube Accuracy: {best_acc:.2f}%\n")
    return best_acc

def run_layer_analysis():
    print("Loading Text Embeddings...")
    clap_model = CLAP(version='2023', use_cuda=False)
    prompts = [f"A person speaking in {lang}" for lang in SEEN_LANGS]
    text_embeddings = clap_model.get_text_embeddings(prompts).detach().to(DEVICE)
    
    print("Loading CLAP Features...")
    clap_train_data = np.load("features_fusion_ekstep.npz")
    clap_test_data = np.load("features_fusion_youtube.npz")
    X_c_train = clap_train_data['X_clap']
    X_c_test = clap_test_data['X_clap']
    y_train = clap_train_data['y']
    y_test = clap_test_data['y']
    
    print("Loading Whisper Multi-Layer Features...")
    w_train_data = np.load("whisper_layers_ekstep.npz")
    w_test_data = np.load("whisper_layers_youtube.npz")
    
    results = {}
    
    # Test Layer 2 (Early Acoustic)
    acc2 = evaluate_layer("Layer 2", X_c_train, w_train_data['X_layer2'], y_train, 
                          X_c_test, w_test_data['X_layer2'], y_test, text_embeddings)
    results["Layer 2"] = acc2
    
    # Test Layer 4 (Middle Phonetic)
    acc4 = evaluate_layer("Layer 4", X_c_train, w_train_data['X_layer4'], y_train, 
                          X_c_test, w_test_data['X_layer4'], y_test, text_embeddings)
    results["Layer 4"] = acc4
    
    # Test Layer 6 (Late Semantic/Phonetic)
    acc6 = evaluate_layer("Layer 6", X_c_train, w_train_data['X_layer6'], y_train, 
                          X_c_test, w_test_data['X_layer6'], y_test, text_embeddings)
    results["Layer 6"] = acc6
    
    print("="*50)
    print("FINAL LAYER-WISE ANALYSIS RESULTS")
    print("="*50)
    for layer, acc in results.items():
        print(f"Whisper {layer}: {acc:.2f}% Accuracy")

if __name__ == "__main__":
    run_layer_analysis()
