"""
Debug script: show what GPT-2 actually generates for 5 samples per language.
Run this AFTER training generative_lid_zero_shot.py to understand the 0% unseen issue.
"""
import torch
import torch.nn as nn
import numpy as np
import random
from transformers import AutoModelForCausalLM, AutoTokenizer

class GenerativeLID(nn.Module):
    def __init__(self, audio_dim, llm_name='gpt2'):
        super().__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(llm_name)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.llm = AutoModelForCausalLM.from_pretrained(llm_name)
        llm_dim = self.llm.config.n_embd
        for param in self.llm.parameters():
            param.requires_grad = False
        self.audio_projection = nn.Sequential(
            nn.Linear(audio_dim, llm_dim),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(llm_dim, llm_dim)
        )

    def generate(self, audio_features, max_new_tokens=15):
        self.eval()
        with torch.no_grad():
            audio_prefix = self.audio_projection(audio_features).unsqueeze(1)
            generated_ids = []
            current_embeds = audio_prefix
            for _ in range(max_new_tokens):
                outputs = self.llm(inputs_embeds=current_embeds)
                next_token_logits = outputs.logits[:, -1, :]
                next_token_id = torch.argmax(next_token_logits, dim=-1)
                generated_ids.append(next_token_id.item())
                if next_token_id.item() == self.tokenizer.eos_token_id:
                    break
                next_embed = self.llm.transformer.wte(next_token_id).unsqueeze(1)
                current_embeds = torch.cat([current_embeds, next_embed], dim=1)
            return self.tokenizer.decode(generated_ids, skip_special_tokens=True)

def debug():
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using device: {device}\n")

    data = np.load("clap_features.npz")
    X, y = data['X'], data['y']

    unique_langs = sorted(list(set(y)))
    random.seed(42)
    langs_copy = unique_langs.copy()
    random.shuffle(langs_copy)
    split_idx = int(0.7 * len(langs_copy))
    seen_langs = langs_copy[:split_idx]
    unseen_langs = langs_copy[split_idx:]

    print(f"Seen: {seen_langs}")
    print(f"Unseen: {unseen_langs}\n")

    # ── Train model (same as generative_lid_zero_shot.py) ──
    seen_mask = np.array([label in seen_langs for label in y])
    X_seen = X[seen_mask]
    y_seen = y[seen_mask]

    audio_dim = X_seen.shape[1]
    model = GenerativeLID(audio_dim=audio_dim).to(device)
    optimizer = torch.optim.AdamW(model.audio_projection.parameters(), lr=5e-4)

    X_tr = torch.tensor(X_seen, dtype=torch.float32).to(device)
    text_targets = [f"The language is {label}." for label in y_seen]
    encoded = model.tokenizer(text_targets, padding=True, return_tensors="pt")
    input_ids = encoded['input_ids'].to(device)
    attention_mask = encoded['attention_mask'].to(device)

    print("Training (5 epochs for debug speed)...")
    for epoch in range(5):
        model.train()
        perm = torch.randperm(X_tr.size(0))
        total_loss = 0
        num_batches = 0
        for i in range(0, X_tr.size(0), 256):
            idx = perm[i:i+256]
            b_audio = X_tr[idx]
            b_ids = input_ids[idx]
            b_mask = attention_mask[idx]

            audio_prefix = model.audio_projection(b_audio).unsqueeze(1)
            text_embeds = model.llm.transformer.wte(b_ids)
            fused = torch.cat([audio_prefix, text_embeds], dim=1)
            prefix_mask = torch.ones((b_audio.size(0), 1), device=device)
            ext_mask = torch.cat([prefix_mask, b_mask], dim=1)
            outputs = model.llm(inputs_embeds=fused, attention_mask=ext_mask)

            shift_logits = outputs.logits[:, :-1, :].contiguous()
            labels = b_ids.clone()
            labels[b_mask == 0] = -100
            loss_fct = nn.CrossEntropyLoss(ignore_index=-100)
            loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), labels.view(-1))
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            total_loss += loss.item()
            num_batches += 1
        print(f"  Epoch {epoch+1}/5 | Loss: {total_loss/num_batches:.4f}")

    # ── SHOW what GPT-2 actually generates ──
    print("\n" + "="*60)
    print("WHAT DOES GPT-2 ACTUALLY GENERATE?")
    print("="*60)

    for lang_list, label in [(seen_langs, "SEEN"), (unseen_langs, "UNSEEN")]:
        print(f"\n--- {label} LANGUAGES ---")
        for lang in lang_list:
            mask = y == lang
            X_lang = X[mask]
            samples = min(5, len(X_lang))
            idxs = np.random.choice(len(X_lang), samples, replace=False)
            print(f"\n  [{lang}] (Expected: 'The language is {lang}.')")
            for j, idx in enumerate(idxs):
                feat = torch.tensor(X_lang[idx:idx+1], dtype=torch.float32).to(device)
                generated = model.generate(feat, max_new_tokens=15)
                match = "✓" if lang.lower() in generated.lower() else "✗"
                print(f"    Sample {j+1}: '{generated}' {match}")

    # ── Diagnose: what does GPT-2 generate from a RANDOM audio prefix (no training)? ──
    print("\n" + "="*60)
    print("SANITY CHECK: Random audio → GPT-2 (no training context)")
    random_feat = torch.randn(1, audio_dim).to(device)
    generated = model.generate(random_feat, max_new_tokens=15)
    print(f"  Random input generates: '{generated}'")

if __name__ == "__main__":
    debug()
