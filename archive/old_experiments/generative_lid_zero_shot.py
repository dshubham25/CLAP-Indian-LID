import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from sklearn.model_selection import train_test_split
from collections import defaultdict
import random

# ==========================================
# GENERATIVE LID NETWORK (Audio Prefix-Tuning)
# ==========================================
class GenerativeLID(nn.Module):
    def __init__(self, audio_dim, llm_name='gpt2'):
        super().__init__()
        
        # 1. Initialize the Frozen LLM and Tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(llm_name)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        
        self.llm = AutoModelForCausalLM.from_pretrained(llm_name)
        llm_dim = self.llm.config.n_embd # 768 for standard GPT-2
        
        # Freeze LLM weights to prevent catastrophic forgetting of language structure
        for param in self.llm.parameters():
            param.requires_grad = False
            
        # 2. The Projection Network (Trainable)
        # Maps the 512D CLAP audio vector into the 768D text embedding space
        self.audio_projection = nn.Sequential(
            nn.Linear(audio_dim, llm_dim),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(llm_dim, llm_dim)
        )
            
    def forward(self, audio_features, input_ids, attention_mask):
        batch_size = audio_features.size(0)
        
        # Project audio and add the sequence dimension: (Batch, 1, 768)
        audio_prefix = self.audio_projection(audio_features).unsqueeze(1)
        
        # Retrieve the standard embeddings for the text tokens from the LLM
        text_embeds = self.llm.transformer.wte(input_ids) # (Batch, Seq_Len, 768)
        
        # Concatenate: [Audio Token, Text Token 1, Text Token 2, ...]
        fused_embeds = torch.cat([audio_prefix, text_embeds], dim=1)
        
        # Extend the attention mask to account for the newly injected audio token
        prefix_mask = torch.ones((batch_size, 1), device=audio_features.device)
        extended_mask = torch.cat([prefix_mask, attention_mask], dim=1)
        
        # Forward pass through the LLM
        outputs = self.llm(inputs_embeds=fused_embeds, attention_mask=extended_mask)
        
        # ==========================================
        # Causal Language Modeling Loss Calculation
        # ==========================================
        # The model predicts token (t) based on tokens (0 to t-1)
        # We shift the logits by 1 to align the predictions with the true labels
        shift_logits = outputs.logits[:, :-1, :].contiguous()
        
        # Mask padding tokens with -100 in labels
        shift_labels = input_ids.clone()
        shift_labels[attention_mask == 0] = -100
        shift_labels = shift_labels.contiguous()
        
        loss_fct = nn.CrossEntropyLoss()
        loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
        
        return loss

    # INCREASED to 10 tokens to allow longer words like 'Malayalam' to finish
    def generate(self, audio_features, max_new_tokens=10):
        """Inference mode: feeds the audio token and lets the LLM speak."""
        self.eval()
        with torch.no_grad():
            audio_prefix = self.audio_projection(audio_features).unsqueeze(1)
            
            # Start generation natively from the audio prefix
            generated_ids = []
            current_embeds = audio_prefix
            
            for _ in range(max_new_tokens):
                outputs = self.llm(inputs_embeds=current_embeds)
                next_token_logits = outputs.logits[:, -1, :]
                next_token_id = torch.argmax(next_token_logits, dim=-1)
                
                generated_ids.append(next_token_id.item())
                
                if next_token_id.item() == self.tokenizer.eos_token_id:
                    break
                    
                # Append predicted token to sequence and continue
                next_embed = self.llm.transformer.wte(next_token_id).unsqueeze(1)
                current_embeds = torch.cat([current_embeds, next_embed], dim=1)
                
            return self.tokenizer.decode(generated_ids, skip_special_tokens=True)


# ==========================================
# TRAINING PIPELINE
# ==========================================
def train_zero_shot(features_file="clap_features.npz"):
    # Target MPS for Apple Silicon Mac
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Training on device: {device}\n")
    
    if not os.path.exists(features_file):
        print("Error: clap_features.npz not found. Run the extraction script first.")
        return
        
    print("Loading Extracted CLAP Features...")
    data = np.load(features_file)
    X, y = data['X'], data['y']
    
    # 1. LANGUAGE SPLIT (8 Seen, 4 Unseen)
    unique_langs = sorted(list(set(y)))
    random.seed(42)
    langs_copy = unique_langs.copy()
    random.shuffle(langs_copy)
    split_idx = int(0.7 * len(langs_copy))
    seen_langs = langs_copy[:split_idx]
    unseen_langs = langs_copy[split_idx:]
    
    print(f"Seen Languages ({len(seen_langs)}): {seen_langs}")
    print(f"Unseen Languages ({len(unseen_langs)}): {unseen_langs}")
    
    # Filter data
    seen_mask = np.isin(y, seen_langs)
    unseen_mask = np.isin(y, unseen_langs)
    
    X_seen, y_seen = X[seen_mask], y[seen_mask]
    X_unseen, y_unseen = X[unseen_mask], y[unseen_mask]
    
    # 70/30 Split on Seen Languages for training/validation
    X_train, X_val_seen, y_train, y_val_seen = train_test_split(
        X_seen, y_seen, test_size=0.3, random_state=42, stratify=y_seen
    )
    
    # Initialize Model
    audio_dim = X.shape[1]
    model = GenerativeLID(audio_dim=audio_dim).to(device)
    
    # Only the projection layer requires gradients
    optimizer = optim.AdamW(model.audio_projection.parameters(), lr=5e-4, weight_decay=1e-4)
    
    print("\n--- Training the Audio Prefix Layer on SEEN Languages ---")
    batch_size = 128
    epochs = 15
    
    # Convert audio features to tensors
    X_tr = torch.tensor(X_train, dtype=torch.float32).to(device)
    
    # Prepare Text Targets: The LLM is trained to output exactly: "The language is [Language]."
    text_targets = [f"The language is {label}." for label in y_train]
    encoded_texts = model.tokenizer(text_targets, padding=True, return_tensors="pt")
    input_ids = encoded_texts['input_ids'].to(device)
    attention_mask = encoded_texts['attention_mask'].to(device)
    
    for epoch in range(epochs):
        model.train()
        permutation = torch.randperm(X_tr.size()[0])
        total_loss = 0
        num_batches = 0
        
        # Batch Processing
        for i in range(0, X_tr.size()[0], batch_size):
            indices = permutation[i:i+batch_size]
            b_audio = X_tr[indices]
            b_input_ids = input_ids[indices]
            b_mask = attention_mask[indices]
            
            optimizer.zero_grad()
            loss = model(b_audio, b_input_ids, b_mask)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
            
        print(f"Epoch {epoch+1:02d}/{epochs} | Language Modeling Loss: {total_loss/num_batches:.4f}")

    def evaluate(X_eval_np, y_eval_np, mode_name):
        print(f"\n--- Evaluating on {mode_name} Languages ---")
        model.eval()
        
        eval_size = min(1000, len(X_eval_np))
        np.random.seed(42)
        indices = np.random.choice(len(X_eval_np), eval_size, replace=False)
        
        X_eval = torch.tensor(X_eval_np[indices], dtype=torch.float32).to(device)
        y_eval_true = y_eval_np[indices]
        
        exact_correct = 0
        sub_correct = 0
        
        lang_exact_correct = defaultdict(int)
        lang_sub_correct = defaultdict(int)
        lang_total = defaultdict(int)
        
        for i in tqdm(range(eval_size), desc=f"Generating ({mode_name})"):
            generated_text = model.generate(X_eval[i:i+1], max_new_tokens=10)
            true_label = y_eval_true[i]
            lang_total[true_label] += 1
            
            expected_text = f"The language is {true_label}."
            
            # Exact match (stripped)
            if generated_text.strip().lower() == expected_text.lower():
                exact_correct += 1
                lang_exact_correct[true_label] += 1
                
            # Substring match
            if true_label.lower() in generated_text.lower():
                sub_correct += 1
                lang_sub_correct[true_label] += 1
                
        exact_acc = exact_correct / eval_size
        sub_acc = sub_correct / eval_size
        
        print(f"  Exact Match Accuracy: {exact_acc * 100:.2f}%")
        print(f"  Substring Match Accuracy: {sub_acc * 100:.2f}%")
        
        per_lang_res = ""
        for lang in sorted(lang_total.keys()):
            l_exact = lang_exact_correct[lang] / lang_total[lang] * 100
            l_sub = lang_sub_correct[lang] / lang_total[lang] * 100
            per_lang_res += f"  {lang}: Exact {l_exact:.2f}% | Substring {l_sub:.2f}%\n"
            print(f"    {lang}: Exact {l_exact:.2f}% | Substring {l_sub:.2f}%")
            
        return exact_acc, sub_acc, per_lang_res

    # Run evaluations
    seen_exact, seen_sub, seen_res = evaluate(X_val_seen, y_val_seen, "SEEN")
    unseen_exact, unseen_sub, unseen_res = evaluate(X_unseen, y_unseen, "UNSEEN")
    
    # Save results
    with open("generative_lid_zero_shot_results.txt", "w") as f:
        f.write("GENERATIVE LID ZERO-SHOT EXPERIMENT RESULTS\n")
        f.write("===========================================\n\n")
        f.write(f"Seen Languages: {seen_langs}\n")
        f.write(f"Unseen Languages: {unseen_langs}\n\n")
        
        f.write("SEEN LANGUAGES ACCURACY:\n")
        f.write(f"Overall Exact Match: {seen_exact * 100:.2f}%\n")
        f.write(f"Overall Substring Match: {seen_sub * 100:.2f}%\n")
        f.write("Per-language breakdown:\n")
        f.write(seen_res)
        f.write("\n")
        
        f.write("UNSEEN LANGUAGES ACCURACY:\n")
        f.write(f"Overall Exact Match: {unseen_exact * 100:.2f}%\n")
        f.write(f"Overall Substring Match: {unseen_sub * 100:.2f}%\n")
        f.write("Per-language breakdown:\n")
        f.write(unseen_res)
        f.write("\n")
        
        f.write("COMPARISON:\n")
        f.write("KNN achieves 0% on unseen, CLAP zero-shot achieves 7.63%, Contrastive Adapter achieves 33.95%\n")
        
    print("\nResults saved to generative_lid_zero_shot_results.txt")

if __name__ == "__main__":
    train_zero_shot()
