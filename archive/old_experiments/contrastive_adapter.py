import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import random
from msclap import CLAP
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

FOLDER_TO_LANG = {
    "hin": "Hindi", "ben": "Bengali", "mar": "Marathi", "tel": "Telugu",
    "tam": "Tamil", "guj": "Gujarati", "urd": "Urdu", "kan": "Kannada",
    "ori": "Odia", "mal": "Malayalam", "pun": "Punjabi", "asm": "Assamese",
}

# 1. Define the Neural Network Adapter
class AudioTextAdapter(nn.Module):
    def __init__(self, input_dim=1024, hidden_dim=512, output_dim=1024):
        super(AudioTextAdapter, self).__init__()
        # A lightweight Multi-Layer Perceptron (MLP) to project audio features
        self.audio_projection = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, output_dim)
        )
        
        # A similar MLP for text to project it into the new joint space
        self.text_projection = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, output_dim)
        )
        
        # Learnable temperature parameter for contrastive loss
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))

    def forward(self, audio_features, text_features):
        audio_proj = self.audio_projection(audio_features)
        text_proj = self.text_projection(text_features)
        
        # Normalize the projected vectors
        audio_proj = F.normalize(audio_proj, p=2, dim=-1)
        text_proj = F.normalize(text_proj, p=2, dim=-1)
        
        return audio_proj, text_proj

def run_contrastive_adapter(features_file="clap_features.npz"):
    # Set device (MPS for Mac Apple Silicon, else CPU)
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using device: {device}")

    print(f"Loading extracted features from {features_file}...")
    data = np.load(features_file)
    X = data['X']  # 512D Audio Embeddings
    y = data['y']  # Labels
    
    unique_langs = sorted(list(set(y)))
    
    # 70% Seen / 30% Unseen Split
    random.seed(42)
    langs_copy = unique_langs.copy()
    random.shuffle(langs_copy)
    split_idx = int(0.7 * len(langs_copy))
    
    seen_langs = langs_copy[:split_idx]
    unseen_langs = langs_copy[split_idx:]
    
    print(f"\nSeen Languages (Training): {seen_langs}")
    print(f"Unseen Languages (Zero-Shot Testing): {unseen_langs}")

    # Generate the Ground Truth Text Embeddings using MS-CLAP
    print("\nGenerating Text Embeddings for the Adapter...")
    clap_model = CLAP(version='2023', use_cuda=False)
    
    all_prompts = [f"A person speaking in {lang}" for lang in unique_langs]
    raw_text_embeddings = clap_model.get_text_embeddings(all_prompts)
    
    # Create a dictionary for quick lookup
    lang_to_text_idx = {lang: i for i, lang in enumerate(unique_langs)}
    text_emb_dict = {lang: raw_text_embeddings[i].clone().detach() for i, lang in enumerate(unique_langs)}

    # Filter data into Seen vs Unseen sets
    seen_mask = np.array([label in seen_langs for label in y])
    unseen_mask = np.array([label in unseen_langs for label in y])
    
    X_seen, y_seen = X[seen_mask], y[seen_mask]
    X_unseen, y_unseen = X[unseen_mask], y[unseen_mask]
    
    # Convert training data to PyTorch Tensors
    X_train_tensor = torch.tensor(X_seen, dtype=torch.float32)
    # The target for the contrastive loss is the index of the correct text prompt
    y_train_indices = torch.tensor([seen_langs.index(label) for label in y_seen], dtype=torch.long)
    
    # Stack the target text embeddings for the seen languages
    seen_text_tensor = torch.stack([text_emb_dict[lang] for lang in seen_langs]).to(device)

    # Initialize Model, Loss, and Optimizer
    input_dimension = X_train_tensor.shape[1] # Should be 512 or 1024 depending on CLAP version
    model = AudioTextAdapter(input_dim=input_dimension, hidden_dim=512, output_dim=input_dimension).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-3)
    criterion = nn.CrossEntropyLoss()

    print("\n--- Training the Contrastive Adapter ---")
    batch_size = 256
    epochs = 20
    model.train()
    
    for epoch in range(epochs):
        permutation = torch.randperm(X_train_tensor.size()[0])
        total_loss = 0
        num_batches = 0
        
        for i in range(0, X_train_tensor.size()[0], batch_size):
            indices = permutation[i:i+batch_size]
            batch_audio = X_train_tensor[indices].to(device)
            batch_labels = y_train_indices[indices].to(device)
            
            optimizer.zero_grad()
            
            # Project audio and ALL seen text prompts into the joint space
            proj_audio, proj_text = model(batch_audio, seen_text_tensor)
            
            # InfoNCE Contrastive Loss logic
            logit_scale = model.logit_scale.exp()
            logits = logit_scale * (proj_audio @ proj_text.T)
            
            loss = criterion(logits, batch_labels)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
            
        if (epoch + 1) % 5 == 0:
            print(f"Epoch {epoch+1}/{epochs} | Loss: {total_loss/num_batches:.4f}")

    print("\n--- Evaluating Zero-Shot Generalization on Unseen Languages ---")
    model.eval()
    with torch.no_grad():
        X_unseen_tensor = torch.tensor(X_unseen, dtype=torch.float32).to(device)
        unseen_text_tensor = torch.stack([text_emb_dict[lang] for lang in unseen_langs]).to(device)
        
        # Project unseen audio and unseen text through the trained adapter
        proj_unseen_audio, proj_unseen_text = model(X_unseen_tensor, unseen_text_tensor)
        
        # Calculate Cosine Similarity
        similarity = (proj_unseen_audio @ proj_unseen_text.T)
        predictions = torch.argmax(similarity, dim=1).cpu().numpy()
        
        # Calculate true indices relative to the unseen languages list
        y_unseen_indices = [unseen_langs.index(label) for label in y_unseen]
        
        acc = accuracy_score(y_unseen_indices, predictions)
        print(f"Adapter Accuracy on UNSEEN Languages: {acc * 100:.2f}%")
        print("(Compare this to the 0.00% of the KNN and the ~7% of Base CLAP!)")

if __name__ == "__main__":
    run_contrastive_adapter()