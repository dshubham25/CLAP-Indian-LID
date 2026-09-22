import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from sklearn.model_selection import train_test_split

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
        
        # Create labels with padding masked out
        labels = input_ids.clone()
        labels[attention_mask == 0] = -100  # Ignore padding in loss
        shift_labels = labels.contiguous()
        
        loss_fct = nn.CrossEntropyLoss(ignore_index=-100)
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
def train_generative_lid(features_file="clap_features.npz"):
    # Target MPS for Apple Silicon Mac
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Training on device: {device}\n")
    
    if not os.path.exists(features_file):
        print("Error: clap_features.npz not found. Run the extraction script first.")
        return
        
    print("Loading Extracted CLAP Features...")
    data = np.load(features_file)
    X, y = data['X'], data['y']
    
    # 70/30 Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
    
    # Initialize Model
    audio_dim = X.shape[1]
    model = GenerativeLID(audio_dim=audio_dim).to(device)
    
    # Only the projection layer requires gradients
    optimizer = optim.AdamW(model.audio_projection.parameters(), lr=5e-4, weight_decay=1e-4)
    
    print("\n--- Training the Audio Prefix Layer ---")
    batch_size = 128
    epochs = 15
    
    # Convert audio features to tensors
    X_tr = torch.tensor(X_train, dtype=torch.float32).to(device)
    
    # Prepare Text Targets
    # The LLM is trained to output exactly: "The language is [Language]."
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

    # ==========================================
    # NEW: Official Accuracy Metric Calculation
    # ==========================================
    print("\n--- Calculating Final Generative Accuracy ---")
    model.eval()
    
    # Take a random subset of 1000 files to keep evaluation fast but statistically significant
    eval_size = min(1000, len(X_test))
    np.random.seed(42) # For reproducibility
    indices = np.random.choice(len(X_test), eval_size, replace=False)
    
    X_eval = torch.tensor(X_test[indices], dtype=torch.float32).to(device)
    y_eval_true = y_test[indices]
    
    correct = 0
    for i in tqdm(range(eval_size), desc="LLM Generating"):
        # The LLM generates the text directly from the audio tensor
        generated_text = model.generate(X_eval[i:i+1], max_new_tokens=10)
        
        # Check if the true language string exists anywhere in the LLM's generated sentence
        if y_eval_true[i].lower() in generated_text.lower():
            correct += 1
            
    final_accuracy = correct / eval_size
    print(f"\n======================================")
    print(f"Generative LID Accuracy: {final_accuracy * 100:.2f}%")
    print(f"======================================")

if __name__ == "__main__":
    train_generative_lid()