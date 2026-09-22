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
from prompts import EKSTEP_PROMPTS

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
class PALM_CLAP(nn.Module):
    def __init__(self, classnames, device='mps'):
        super().__init__()
        self.clap = CLAP(version='2023', use_cuda=False)
        self.clap.read_audio = custom_read_audio.__get__(self.clap, CLAP)
        self.classnames = classnames
        
        # Get fixed text embeddings on CPU first to avoid MPS tensor mismatch in msclap
        print("Extracting fixed text embeddings...")
        with torch.no_grad():
            prompts = [f"This is a person speaking in {lang}." for lang in classnames]
            self.fixed_text_features = self.clap.get_text_embeddings(prompts).to(device) # [num_classes, 1024]

        self.clap.clap = self.clap.clap.to(device)
        self.device = device
        
        # Freeze CLAP
        for param in self.clap.clap.parameters():
            param.requires_grad = False
            
        # PALM: Prompt Learning in the Audio-Language Feature Space
        # Instead of going through the 12-layer text encoder, we learn a projection
        # in the 1024D joint space that shifts the fixed text embeddings.
        self.prompt_adapter = nn.Sequential(
            nn.Linear(1024, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 1024)
        )
        # Initialize adapter to identity (zero shift) initially
        nn.init.zeros_(self.prompt_adapter[-1].weight)
        nn.init.zeros_(self.prompt_adapter[-1].bias)
        
        # Temperature
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))

    def forward(self, audio_files):
        # Process audio manually to ensure it moves to the correct device
        audio_tensors = []
        for f in audio_files:
            a, _ = self.clap.read_audio(f)
            audio_tensors.append(a.squeeze(0))
            
        max_audio_len = max([a.size(0) for a in audio_tensors])
        padded_audio = []
        for a in audio_tensors:
            pad_amount = max_audio_len - a.size(0)
            padded_audio.append(F.pad(a, (0, pad_amount)))
        audio_batch = torch.stack(padded_audio).to(self.device)
        
        # Get raw audio embeddings
        with torch.no_grad():
            audio_features = self.clap.clap.audio_encoder(audio_batch)[0] # [batch, 1024]
            
        # Apply PALM adapter to the text features
        # text_features: [num_classes, 1024]
        shifted_text_features = self.fixed_text_features + self.prompt_adapter(self.fixed_text_features)
        
        # Normalize
        audio_features = F.normalize(audio_features, dim=-1)
        shifted_text_features = F.normalize(shifted_text_features, dim=-1)
        
        # Cosine similarity
        logit_scale = self.logit_scale.exp()
        logits = logit_scale * audio_features @ shifted_text_features.t()
        
        return logits

def train(epochs=10, batch_size=32):
    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    train_dataset = AudioDataset("ekstep_seen", SEEN_LANGS)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    
    model = PALM_CLAP(classnames=SEEN_LANGS, device=device).to(device)
    
    optimizer = torch.optim.AdamW([
        {'params': model.prompt_adapter.parameters()},
        {'params': [model.logit_scale]}
    ], lr=1e-3, weight_decay=1e-4)
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
            
    # Save the adapter
    torch.save(model.prompt_adapter.state_dict(), "palm_adapter.pt")
    print("Training complete. Saved to palm_adapter.pt")

if __name__ == "__main__":
    train(epochs=5)
