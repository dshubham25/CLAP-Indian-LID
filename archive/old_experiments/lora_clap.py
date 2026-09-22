import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random
import os
import argparse
import glob
import soundfile as sf
from tqdm import tqdm
from transformers import AutoProcessor, ClapModel
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from torch.utils.data import Dataset, DataLoader

# -----------------------------------------------------------------------------
# LoRA Implementation for End-to-End Mode
# -----------------------------------------------------------------------------
class LoRALayer(nn.Module):
    """Low-Rank Adaptation layer.
    Adds a low-rank decomposition BA to the frozen weight matrix W.
    Output = Wx + (B @ A)x * (alpha / rank)
    """
    def __init__(self, original_layer, rank=8, alpha=16):
        super().__init__()
        self.original_layer = original_layer
        
        if isinstance(original_layer, nn.Linear):
            in_features = original_layer.in_features
            out_features = original_layer.out_features
        else:
            raise ValueError(f"Unsupported layer type: {type(original_layer)}")
            
        # Freeze original weights
        for param in self.original_layer.parameters():
            param.requires_grad = False
            
        # Low-rank matrices
        self.lora_A = nn.Parameter(torch.randn(in_features, rank) * 0.01)
        self.lora_B = nn.Parameter(torch.zeros(rank, out_features))
        self.scaling = alpha / rank
        
    def forward(self, x):
        original_output = self.original_layer(x)
        lora_output = (x @ self.lora_A @ self.lora_B) * self.scaling
        return original_output + lora_output

def apply_lora_to_model(model, rank=8, alpha=16, target_modules=['q_proj', 'v_proj']):
    """Apply LoRA to specified linear layers in the audio encoder."""
    lora_params = []
    replaced = 0
    
    # We only want to apply LoRA to the audio encoder
    if hasattr(model, 'audio_model'):
        target_model = model.audio_model
    else:
        target_model = model
    
    # Create a list of tuples (name, module) to iterate over since we'll modify the model
    module_list = [(n, m) for n, m in target_model.named_modules() if isinstance(m, nn.Linear)]
    
    for name, module in module_list:
        # Check if this module should get LoRA
        if any(target in name for target in target_modules):
            # Find parent module and attribute name
            parts = name.split('.')
            parent = target_model
            for part in parts[:-1]:
                parent = getattr(parent, part)
            attr_name = parts[-1]
            
            # Replace with LoRA layer
            lora_layer = LoRALayer(module, rank=rank, alpha=alpha)
            setattr(parent, attr_name, lora_layer)
            lora_params.extend([lora_layer.lora_A, lora_layer.lora_B])
            replaced += 1
            
    print(f"Applied LoRA to {replaced} layers")
    total_lora_params = sum(p.numel() for p in lora_params)
    total_model_params = sum(p.numel() for p in model.parameters())
    print(f"LoRA parameters: {total_lora_params:,} ({100*total_lora_params/total_model_params:.2f}% of total)")
    
    return lora_params

# -----------------------------------------------------------------------------
# LoRA Adapter for Features Mode
# -----------------------------------------------------------------------------
class LoRAAdapter(nn.Module):
    """LoRA-style adapter applied to pre-extracted features.
    Includes a projection to match text embedding dimensionality (512D).
    """
    def __init__(self, feature_dim, text_dim=512, rank=8, alpha=16):
        super().__init__()
        self.lora_A = nn.Parameter(torch.randn(feature_dim, rank) * 0.01)
        self.lora_B = nn.Parameter(torch.zeros(rank, feature_dim))
        self.scaling = alpha / rank
        # Project from feature_dim (1024) to text_dim (512) for contrastive alignment
        self.projection = nn.Linear(feature_dim, text_dim, bias=False)
        
    def forward(self, x):
        # x + low-rank residual, then project to text embedding space
        adapted = x + (x @ self.lora_A @ self.lora_B) * self.scaling
        return self.projection(adapted)

# -----------------------------------------------------------------------------
# Datasets
# -----------------------------------------------------------------------------
class AudioDataset(Dataset):
    """Dataset for end-to-end processing from raw audio."""
    def __init__(self, audio_paths, labels, processor, max_duration=10.0, sample_rate=48000):
        self.audio_paths = audio_paths
        self.labels = labels
        self.processor = processor
        self.max_length = int(max_duration * sample_rate)
        
    def __len__(self):
        return len(self.audio_paths)
        
    def __getitem__(self, idx):
        path = self.audio_paths[idx]
        label = self.labels[idx]
        
        try:
            # Load audio using soundfile
            audio_array, sr = sf.read(path)
            
            # Ensure mono
            if len(audio_array.shape) > 1:
                audio_array = audio_array.mean(axis=1)
                
            # Truncate or pad to max_length
            if len(audio_array) > self.max_length:
                audio_array = audio_array[:self.max_length]
            elif len(audio_array) < self.max_length:
                padding = np.zeros(self.max_length - len(audio_array))
                audio_array = np.concatenate([audio_array, padding])
                
            return {"audio": audio_array, "label": label, "sr": sr}
        except Exception as e:
            # Fallback for corrupted files
            print(f"Error loading {path}: {e}")
            return {"audio": np.zeros(self.max_length), "label": label, "sr": 48000}

class FeaturesDataset(Dataset):
    """Dataset for pre-extracted features."""
    def __init__(self, features, labels):
        self.features = torch.tensor(features, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.long)
        
    def __len__(self):
        return len(self.features)
        
    def __getitem__(self, idx):
        return {"features": self.features[idx], "label": self.labels[idx]}

# -----------------------------------------------------------------------------
# Main Implementation
# -----------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="LoRA CLAP for Indian Language ID")
    parser.add_argument("--mode", type=str, choices=["end2end", "features"], default="features",
                        help="Training mode: end2end (raw audio) or features (pre-extracted)")
    parser.add_argument("--rank", type=int, default=8, help="LoRA rank")
    parser.add_argument("--alpha", type=float, default=16.0, help="LoRA alpha scaling factor")
    parser.add_argument("--epochs", type=int, default=20, help="Number of training epochs")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--batch-size", type=int, default=256, help="Batch size")
    parser.add_argument("--features-file", type=str, default="clap_features.npz",
                        help="Path to pre-extracted features (for features mode)")
    parser.add_argument("--dataset-root", type=str, default="ekstep_seen",
                        help="Root directory for audio files (for end2end mode)")
    parser.add_argument("--results-file", type=str, default="lora_results.txt",
                        help="File to save results")
    args = parser.parse_args()
    
    # Device setup
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    print(f"Using device: {device}")
    
    # Setup Data & Labels
    print("Preparing dataset and language splits...")
    if args.mode == "features":
        if not os.path.exists(args.features_file):
            print(f"Error: Features file {args.features_file} not found.")
            return
            
        data = np.load(args.features_file)
        X = data['X']          # shape: (N, 1024)
        lang_names = data['y'] # shape: (N,) - string language labels
        # Build label index mapping
        unique_langs = sorted(list(set(lang_names)))
        lang2idx = {lang: i for i, lang in enumerate(unique_langs)}
        y = np.array([lang2idx[lang] for lang in lang_names])
    else:
        # Load paths from root dir
        audio_paths = []
        lang_names = []
        for file_path in glob.glob(f"{args.dataset_root}/**/*.wav", recursive=True):
            audio_paths.append(file_path)
            # Assuming format like ekstep_seen/language/file.wav
            lang = os.path.basename(os.path.dirname(file_path))
            lang_names.append(lang)
            
        if not audio_paths:
            print(f"No audio files found in {args.dataset_root}")
            return
            
        unique_langs = sorted(list(set(lang_names)))
        lang2idx = {lang: i for i, lang in enumerate(unique_langs)}
        y = np.array([lang2idx[lang] for lang in lang_names])
        X = np.array(audio_paths)
    
    idx2lang = {i: lang for lang, i in lang2idx.items()}
    
    # Split into seen and unseen languages (70/30 split)
    random.seed(42)
    langs_copy = unique_langs.copy()
    random.shuffle(langs_copy)
    split_idx = int(0.7 * len(langs_copy))
    seen_langs = langs_copy[:split_idx]
    unseen_langs = langs_copy[split_idx:]
    
    print(f"Seen languages ({len(seen_langs)}): {seen_langs}")
    print(f"Unseen languages ({len(unseen_langs)}): {unseen_langs}")
    
    # Filter indices for seen and unseen
    seen_idx = [i for i, lang in enumerate(lang_names) if lang in seen_langs]
    unseen_idx = [i for i, lang in enumerate(lang_names) if lang in unseen_langs]
    
    X_seen, y_seen = X[seen_idx], y[seen_idx]
    X_unseen, y_unseen = X[unseen_idx], y[unseen_idx]
    
    # Further split seen into train/val
    # Stratified split to ensure all seen languages are well represented
    try:
        X_train, X_val, y_train, y_val = train_test_split(
            X_seen, y_seen, test_size=0.2, random_state=42, stratify=y_seen
        )
    except ValueError:
        # Fallback if stratify fails due to low counts
        X_train, X_val, y_train, y_val = train_test_split(
            X_seen, y_seen, test_size=0.2, random_state=42
        )
        
    print(f"Training samples: {len(X_train)}")
    print(f"Validation samples: {len(X_val)}")
    print(f"Zero-shot (unseen) samples: {len(X_unseen)}")
    
    # Prepare text embeddings for contrastive loss
    print("Generating text embeddings for languages...")
    # Load model and processor just to get text embeddings once
    processor = AutoProcessor.from_pretrained("laion/clap-htsat-unfused")
    base_model = ClapModel.from_pretrained("laion/clap-htsat-unfused").to(device)
    base_model.eval()
    
    # Create text descriptions for all languages
    texts = [f"A person speaking in {lang} language." for lang in unique_langs]
    text_inputs = processor(text=texts, padding=True, return_tensors="pt").to(device)
    
    with torch.no_grad():
        text_features = base_model.get_text_features(**text_inputs)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
    
    # Ensure text features mapping matches indices
    # text_features[i] corresponds to idx2lang[i]
    
    # Model Setup
    print(f"Setting up model in {args.mode} mode...")
    
    if args.mode == "end2end":
        # Keep the base model but apply LoRA
        # Freeze ALL parameters
        for param in base_model.parameters():
            param.requires_grad = False
            
        # Apply LoRA to audio encoder
        lora_params = apply_lora_to_model(base_model, rank=args.rank, alpha=args.alpha)
        
        # Optimizer for LoRA parameters only
        optimizer = torch.optim.AdamW(lora_params, lr=args.lr)
        
        # DataLoader
        train_dataset = AudioDataset(X_train, y_train, processor)
        val_dataset = AudioDataset(X_val, y_val, processor)
        unseen_dataset = AudioDataset(X_unseen, y_unseen, processor)
        
        def collate_fn(batch):
            audio = [item["audio"] for item in batch]
            labels = torch.tensor([item["label"] for item in batch], dtype=torch.long)
            # Use processor to process audio batch
            inputs = processor(audios=audio, return_tensors="pt", sampling_rate=48000, padding=True)
            return inputs, labels
            
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn, num_workers=4)
        val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn, num_workers=4)
        unseen_loader = DataLoader(unseen_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn, num_workers=4)
        
        model_to_train = base_model
        
    else:  # features mode
        feature_dim = X.shape[1]
        adapter = LoRAAdapter(feature_dim, rank=args.rank, alpha=args.alpha).to(device)
        optimizer = torch.optim.AdamW(adapter.parameters(), lr=args.lr)
        
        print(f"Features adapter parameters: {sum(p.numel() for p in adapter.parameters()):,}")
        
        train_dataset = FeaturesDataset(X_train, y_train)
        val_dataset = FeaturesDataset(X_val, y_val)
        unseen_dataset = FeaturesDataset(X_unseen, y_unseen)
        
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
        unseen_loader = DataLoader(unseen_dataset, batch_size=args.batch_size, shuffle=False)
        
        model_to_train = adapter

    # Temperature parameter for InfoNCE loss (fixed, not learned — avoids MPS leaf-tensor issues)
    logit_scale = float(np.log(1 / 0.07))  # ~2.659
    
    # Training Loop
    print("Starting training...")
    best_val_acc = 0.0
    
    for epoch in range(args.epochs):
        if args.mode == "features":
            model_to_train.train()
        else:
            # We don't want to turn on batchnorm/dropout in the frozen base model
            model_to_train.eval()
            
        train_loss = 0.0
        
        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs}"):
            optimizer.zero_grad()
            
            if args.mode == "end2end":
                inputs, labels = batch
                inputs = {k: v.to(device) for k, v in inputs.items()}
                labels = labels.to(device)
                
                # Get audio features through LoRA-adapted model
                audio_features = model_to_train.get_audio_features(**inputs)
            else:
                features = batch["features"].to(device)
                labels = batch["label"].to(device)
                
                # Pass features through LoRA adapter
                audio_features = model_to_train(features)
                
            # Normalize audio features
            audio_features = audio_features / audio_features.norm(dim=-1, keepdim=True)
            
            # Select target text features for the batch
            batch_text_features = text_features[labels]
            
            # InfoNCE / Contrastive Loss
            # Cosine similarity matrix between all audios and all texts in batch
            scale = np.exp(logit_scale)
            logits_per_audio = scale * audio_features @ batch_text_features.t()
            logits_per_text = logits_per_audio.t()
            
            # Labels for contrastive loss: diagonal should be highest
            batch_size_cur = labels.shape[0]
            contrastive_labels = torch.arange(batch_size_cur, device=device)
            
            loss_a = F.cross_entropy(logits_per_audio, contrastive_labels)
            loss_t = F.cross_entropy(logits_per_text, contrastive_labels)
            loss = (loss_a + loss_t) / 2
            
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            
        train_loss /= len(train_loader)
        
        # Validation Loop
        if args.mode == "features":
            model_to_train.eval()
            
        val_preds = []
        val_trues = []
        
        with torch.no_grad():
            for batch in val_loader:
                if args.mode == "end2end":
                    inputs, labels = batch
                    inputs = {k: v.to(device) for k, v in inputs.items()}
                    
                    audio_features = model_to_train.get_audio_features(**inputs)
                else:
                    features = batch["features"].to(device)
                    labels = batch["label"]
                    
                    audio_features = model_to_train(features)
                    
                audio_features = audio_features / audio_features.norm(dim=-1, keepdim=True)
                
                # Similarity with ALL text features (for zero-shot style classification)
                logits = audio_features @ text_features.t()
                preds = torch.argmax(logits, dim=1).cpu().numpy()
                
                val_preds.extend(preds)
                val_trues.extend(labels.numpy() if isinstance(labels, torch.Tensor) else labels)
                
        val_acc = accuracy_score(val_trues, val_preds)
        print(f"Epoch {epoch+1} - Loss: {train_loss:.4f} - Val Acc: {val_acc:.4f}")
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            # Save best weights
            if args.mode == "features":
                torch.save(model_to_train.state_dict(), "best_lora_adapter.pt")
            else:
                # Save just the lora params
                lora_state = {name: param.data for name, param in model_to_train.named_parameters() if param.requires_grad}
                torch.save(lora_state, "best_lora_weights.pt")
                
    # Evaluation on Unseen Languages (Zero-Shot)
    print("\nEvaluating on UNSEEN languages (Zero-Shot)...")
    if args.mode == "features":
        model_to_train.load_state_dict(torch.load("best_lora_adapter.pt"))
    else:
        # Note: in end2end we just use the weights since it's already in memory, 
        # normally we'd load the state dict back
        pass
        
    if args.mode == "features":
        model_to_train.eval()
        
    unseen_preds = []
    unseen_trues = []
    
    with torch.no_grad():
        for batch in unseen_loader:
            if args.mode == "end2end":
                inputs, labels = batch
                inputs = {k: v.to(device) for k, v in inputs.items()}
                
                audio_features = model_to_train.get_audio_features(**inputs)
            else:
                features = batch["features"].to(device)
                labels = batch["label"]
                
                audio_features = model_to_train(features)
                
            audio_features = audio_features / audio_features.norm(dim=-1, keepdim=True)
            
            logits = audio_features @ text_features.t()
            preds = torch.argmax(logits, dim=1).cpu().numpy()
            
            unseen_preds.extend(preds)
            unseen_trues.extend(labels.numpy() if isinstance(labels, torch.Tensor) else labels)
            
    unseen_acc = accuracy_score(unseen_trues, unseen_preds)
    print(f"Zero-Shot Accuracy on Unseen Languages: {unseen_acc:.4f}")
    
    # Compute per-language accuracy
    print("\nPer-Language Zero-Shot Accuracy:")
    per_lang_acc = {}
    for lang in unseen_langs:
        lang_idx = lang2idx[lang]
        indices = [i for i, true_label in enumerate(unseen_trues) if true_label == lang_idx]
        if indices:
            lang_trues = [unseen_trues[i] for i in indices]
            lang_preds = [unseen_preds[i] for i in indices]
            acc = accuracy_score(lang_trues, lang_preds)
            per_lang_acc[lang] = acc
            print(f"  {lang}: {acc:.4f} ({len(indices)} samples)")
            
    # Save results
    with open(args.results_file, "a") as f:
        f.write(f"=== LoRA CLAP Configuration ===\n")
        f.write(f"Mode: {args.mode}, Rank: {args.rank}, Alpha: {args.alpha}, Epochs: {args.epochs}, LR: {args.lr}\n")
        f.write(f"Best Seen Validation Accuracy: {best_val_acc:.4f}\n")
        f.write(f"Unseen Zero-Shot Accuracy: {unseen_acc:.4f}\n")
        f.write(f"Per-Language Unseen Accuracy:\n")
        for lang, acc in per_lang_acc.items():
            f.write(f"  {lang}: {acc:.4f}\n")
        f.write(f"--------------------------------------------------\n\n")

if __name__ == "__main__":
    main()
