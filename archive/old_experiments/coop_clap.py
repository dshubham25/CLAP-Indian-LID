import torch
import torch.nn as nn
import numpy as np
from msclap import CLAP
import soundfile as sf
import torchaudio.transforms as T
from torch.utils.data import DataLoader, Dataset
import torch.nn.functional as F
import os
from tqdm import tqdm

# --- MONKEY PATCH ---
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

# --- DATASET ---
FOLDER_TO_LANG = {
    "hin": "Hindi", "ben": "Bengali", "mar": "Marathi", "tel": "Telugu",
    "tam": "Tamil", "guj": "Gujarati", "urd": "Urdu", "kan": "Kannada",
    "ori": "Odia", "odi": "Odia", "mal": "Malayalam", "pun": "Punjabi",
    "asm": "Assamese", "eng": "English"
}
SEEN_LANGS = ['Marathi', 'Kannada', 'English', 'Odia', 'Punjabi', 'Malayalam', 'Telugu', 'Gujarati']
UNSEEN_LANGS = ['Hindi', 'Assamese', 'Bengali', 'Tamil']

class AudioDataset(Dataset):
    def __init__(self, folder_path, langs):
        self.files = []
        self.labels = []
        self.lang_to_idx = {l: i for i, l in enumerate(langs)}
        
        for root, _, files in os.walk(folder_path):
            folder_name = os.path.basename(root)
            if folder_name in FOLDER_TO_LANG:
                lang_name = FOLDER_TO_LANG[folder_name]
                if lang_name in langs:
                    for f in files:
                        if f.endswith('.wav'):
                            self.files.append(os.path.join(root, f))
                            self.labels.append(self.lang_to_idx[lang_name])
                            
    def __len__(self):
        return len(self.files)
        
    def __getitem__(self, idx):
        return self.files[idx], self.labels[idx]

# --- MODEL ---
class CoOpCLAP(nn.Module):
    def __init__(self, n_ctx=4, classnames=None, device='mps'):
        super().__init__()
        self.clap = CLAP(version='2023', use_cuda=False)
        self.clap.read_audio = custom_read_audio.__get__(self.clap, CLAP)
        self.clap.clap = self.clap.clap.to(device)
        self.device = device
        
        # Freeze CLAP
        for param in self.clap.clap.parameters():
            param.requires_grad = False
            
        self.n_ctx = n_ctx
        self.classnames = classnames
        self.ctx_dim = self.clap.clap.caption_encoder.base.config.hidden_size # 768 for GPT2
        
        # Initialize learnable context vectors
        print(f"Initializing CoOp with {n_ctx} learnable tokens of dim {self.ctx_dim}")
        ctx_init = torch.empty(n_ctx, self.ctx_dim)
        nn.init.normal_(ctx_init, std=0.02)
        self.ctx = nn.Parameter(ctx_init)
        
        # Temperature parameter for InfoNCE loss
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
        
        # Pre-tokenize classnames
        tokenizer = self.clap.tokenizer
        tokenized_classes = tokenizer(classnames)
        self.class_input_ids = [torch.tensor(ids).to(device) for ids in tokenized_classes['input_ids']]

    def forward_text(self):
        # We need to construct the text embeddings for all classnames
        batch_size = len(self.classnames)
        
        text_encoder = self.clap.clap.caption_encoder
        word_embeddings = text_encoder.base.wte
        
        ctx = self.ctx.unsqueeze(0).expand(batch_size, -1, -1) # [batch, n_ctx, ctx_dim]
        
        prompts = []
        dummy_input_ids = []
        
        # Max sequence length calculation for padding
        max_len = max([ids.size(0) for ids in self.class_input_ids]) + self.n_ctx
        
        for i, class_ids in enumerate(self.class_input_ids):
            class_embeds = word_embeddings(class_ids).unsqueeze(0) # [1, seq_len, ctx_dim]
            
            # [ctx, class_name]
            prompt_embeds = torch.cat([ctx[i:i+1], class_embeds], dim=1) # [1, n_ctx + seq_len, ctx_dim]
            
            # Create dummy input ids just for the length calculation inside MS-CLAP TextEncoder
            # GPT2 in MS-CLAP uses length to find the last token.
            # MS-CLAP ignores 0s for length. So we use 1 for valid tokens.
            seq_len = prompt_embeds.size(1)
            dummy_ids = torch.ones((1, seq_len), dtype=torch.long, device=self.device)
            
            # Pad if necessary to form a batch
            pad_len = max_len - seq_len
            if pad_len > 0:
                prompt_embeds = F.pad(prompt_embeds, (0, 0, 0, pad_len))
                dummy_ids = F.pad(dummy_ids, (0, pad_len), value=0) # 0 is padding
                
            prompts.append(prompt_embeds)
            dummy_input_ids.append(dummy_ids)
            
        prompts = torch.cat(prompts, dim=0) # [batch, max_len, ctx_dim]
        dummy_input_ids = torch.cat(dummy_input_ids, dim=0) # [batch, max_len]
        
        # Forward through GPT-2 base directly to avoid input_ids conflict
        outputs = text_encoder.base(
            inputs_embeds=prompts, 
            attention_mask=(dummy_input_ids != 0).long()
        )
        hidden_states = outputs[0] # [batch, max_len, 768]
        
        # Get the sequence lengths to extract the final token representation
        sequence_lengths = torch.ne(dummy_input_ids, 0).sum(-1) - 1
        
        # Extract last token
        out = hidden_states[torch.arange(batch_size, device=hidden_states.device), sequence_lengths]
        
        # Project to 1024D
        text_features = text_encoder.projection(out)
        return text_features # [batch, 1024]

    def forward(self, audio_files):
        # Process audio manually to ensure it moves to the correct device
        audio_tensors = []
        for f in audio_files:
            a, _ = self.clap.read_audio(f)
            audio_tensors.append(a.squeeze(0))
            
        # Pad audio tensors to same length
        max_audio_len = max([a.size(0) for a in audio_tensors])
        padded_audio = []
        for a in audio_tensors:
            pad_amount = max_audio_len - a.size(0)
            padded_audio.append(F.pad(a, (0, pad_amount)))
        audio_batch = torch.stack(padded_audio).to(self.device)
        
        with torch.no_grad():
            audio_features = self.clap.clap.audio_encoder(audio_batch)[0] # [batch, 1024]
        
        # Get text embeddings
        text_features = self.forward_text()
        
        # Normalize
        audio_features = F.normalize(audio_features, dim=-1)
        text_features = F.normalize(text_features, dim=-1)
        
        # Cosine similarity
        logit_scale = self.logit_scale.exp()
        logits = logit_scale * audio_features @ text_features.t()
        
        return logits

def train(epochs=10, batch_size=32):
    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    train_dataset = AudioDataset("ekstep_seen", SEEN_LANGS)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    
    model = CoOpCLAP(n_ctx=4, classnames=SEEN_LANGS, device=device).to(device)
    
    # ONLY optimize the prompt vectors and temperature
    optimizer = torch.optim.AdamW([model.ctx, model.logit_scale], lr=1e-3, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()
    
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        correct = 0
        total = 0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")
        for audio_files, labels in pbar:
            labels = labels.to(device)
            
            optimizer.zero_grad()
            logits = model(audio_files) # [batch, num_classes]
            
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            preds = logits.argmax(dim=-1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
            
            pbar.set_postfix({'loss': f"{loss.item():.4f}", 'acc': f"{correct/total:.4f}"})
            
    # Save the learned prompts
    torch.save(model.ctx.detach().cpu(), "coop_learned_prompts.pt")
    print("Training complete. Saved prompts to coop_learned_prompts.pt")

if __name__ == "__main__":
    train(epochs=5)
