import os
import glob
import time
import types
import torch
import torch.nn as nn
import torch.optim as optim
import soundfile as sf
import torchaudio.transforms as T
import numpy as np
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from msclap import CLAP

# Layout mapping for the ekstep_seen dataset folders
FOLDER_TO_LANG = {
    "hin": "Hindi", "ben": "Bengali", "mar": "Marathi", "tel": "Telugu",
    "tam": "Tamil", "guj": "Gujarati", "urd": "Urdu", "kan": "Kannada",
    "ori": "Odia", "odi": "Odia", "mal": "Malayalam", "pun": "Punjabi",
    "asm": "Assamese", "mai": "Maithili", "sat": "Santali", "kas": "Kashmiri",
    "nep": "Nepali", "snd": "Sindhi", "kok": "Konkani", "doi": "Dogri",
    "mni": "Manipuri", "brx": "Bodo", "san": "Sanskrit", "eng": "English",
}

# --- THE MONKEY PATCH (Mac Soundfile Fix) ---
def custom_read_audio(self, audio_path, resample=True):
    audio_array, sample_rate = sf.read(audio_path)
    if len(audio_array.shape) > 1:
        audio_array = audio_array.mean(axis=1) # Mono
        
    audio_tensor = torch.FloatTensor(audio_array)
    if len(audio_tensor.shape) == 1:
        audio_tensor = audio_tensor.unsqueeze(0)
    else:
        audio_tensor = audio_tensor.T
        
    resample_rate = self.args.sampling_rate
    if resample and resample_rate != sample_rate:
        resampler = T.Resample(sample_rate, resample_rate)
        audio_tensor = resampler(audio_tensor)
        
    return audio_tensor, resample_rate


# ==========================================
# 1. PYTORCH DATASET FOR AUDIO FILES
# ==========================================
class EkstepDataset(Dataset):
    def __init__(self, file_paths, labels, lang_to_idx, clap_model, target_duration=7.0):
        self.file_paths = file_paths
        self.labels = labels
        self.lang_to_idx = lang_to_idx
        self.clap_model = clap_model
        self.target_length = int(target_duration * clap_model.args.sampling_rate)

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        file_path = self.file_paths[idx]
        label_str = self.labels[idx]
        label_idx = self.lang_to_idx[label_str]
        
        try:
            # Read audio using soundfile patch
            audio_tensor, sr = self.clap_model.read_audio(file_path, resample=True)
            audio_tensor = audio_tensor.squeeze(0)
            
            # Crop or Pad audio to uniform duration
            if audio_tensor.size(0) > self.target_length:
                audio_tensor = audio_tensor[:self.target_length]
            elif audio_tensor.size(0) < self.target_length:
                padding = self.target_length - audio_tensor.size(0)
                audio_tensor = torch.nn.functional.pad(audio_tensor, (0, padding))
                
        except Exception:
            audio_tensor = torch.zeros(self.target_length)
            
        return audio_tensor, label_idx


# ==========================================
# 2. UNLOCKED CLAP MODEL ARCHITECTURE
# ==========================================
class UnlockedCLAPClassifier(nn.Module):
    def __init__(self, clap_wrapper, num_classes):
        super().__init__()
        self.clap_wrapper = clap_wrapper
        self.underlying_model = getattr(self.clap_wrapper, 'clap', self.clap_wrapper)
        
        # Unfreeze underlying model weights for fine-tuning
        for param in self.underlying_model.parameters():
            param.requires_grad = True

        self.num_classes = num_classes

        # Probe embedding dimension eagerly on CPU during initialization
        with torch.no_grad():
            dummy_audio = torch.zeros(1, 112000) # 7 seconds at 16kHz
            dummy_embed = self._extract_embeddings(dummy_audio)
            embed_dim = dummy_embed.shape[-1]

        # Register classification head during init so optimizer includes its parameters
        self.classifier = nn.Sequential(
            nn.Linear(embed_dim, 512),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(256, self.num_classes)
        )

    def _extract_embeddings(self, audio_tensors):
        """Extracts tensor embeddings and handles tuple outputs from MS-CLAP."""
        clap_module = getattr(self.clap_wrapper, 'clap', self.clap_wrapper)
        
        if hasattr(clap_module, 'audio_encoder'):
            feats = clap_module.audio_encoder(audio_tensors)
            if isinstance(feats, (tuple, list)):
                feats = feats[0]
            if hasattr(clap_module, 'audio_projection'):
                proj = clap_module.audio_projection(feats)
                if isinstance(proj, (tuple, list)):
                    proj = proj[0]
                return proj
            return feats
        elif hasattr(self.clap_wrapper, 'get_audio_embeddings_per_batch'):
            res = self.clap_wrapper.get_audio_embeddings_per_batch(audio_tensors)
            if isinstance(res, (tuple, list)):
                res = res[0]
            return res
        else:
            res = clap_module(audio_tensors)
            if isinstance(res, (tuple, list)):
                res = res[0]
            return res

    def forward(self, audio_tensors):
        embeddings = self._extract_embeddings(audio_tensors)
        return self.classifier(embeddings)


# ==========================================
# 3. TRAINING & EVALUATION PIPELINE
# ==========================================
def run_unlocked_finetune(dataset_root, batch_size=32, epochs=10, lr=1e-4):
    start_time = time.time()
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Executing Unlocked Fine-Tuning on Device: {device}\n")

    # Initialize MS-CLAP
    clap_wrapper = CLAP(version='2023', use_cuda=False)
    clap_wrapper.read_audio = types.MethodType(custom_read_audio, clap_wrapper)

    # Gather audio files
    wav_files = glob.glob(os.path.join(dataset_root, "**", "*.wav"), recursive=True)
    if not wav_files:
        print(f"No .wav files found in {dataset_root}.")
        return

    valid_files, valid_labels = [], []
    for f in wav_files:
        folder = os.path.basename(os.path.dirname(f))
        if folder in FOLDER_TO_LANG:
            valid_files.append(f)
            valid_labels.append(FOLDER_TO_LANG[folder])

    unique_langs = sorted(list(set(valid_labels)))
    lang_to_idx = {lang: i for i, lang in enumerate(unique_langs)}
    num_classes = len(unique_langs)

    print(f"Found {len(valid_files)} valid audio files across {num_classes} languages.")

    # Stratified 70/30 Train/Test Split
    train_files, test_files, train_labels, test_labels = train_test_split(
        valid_files, valid_labels, test_size=0.3, random_state=42, stratify=valid_labels
    )

    # PyTorch DataLoaders
    train_dataset = EkstepDataset(train_files, train_labels, lang_to_idx, clap_wrapper)
    test_dataset = EkstepDataset(test_files, test_labels, lang_to_idx, clap_wrapper)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    # Instantiate Unlocked Model
    model = UnlockedCLAPClassifier(clap_wrapper, num_classes).to(device)
    
    # Paper Hyperparameters: Adam Optimizer + Learning Rate 1e-4
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=lr)
    criterion = nn.CrossEntropyLoss()

    print(f"\n--- Starting End-to-End Fine-Tuning ({epochs} Epochs | LR={lr}) ---")

    best_acc = 0.0
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        
        for audio_batch, label_batch in tqdm(train_loader, desc=f"Epoch {epoch+1:02d}/{epochs}"):
            audio_batch = audio_batch.to(device)
            label_batch = label_batch.to(device)
            
            optimizer.zero_grad()
            logits = model(audio_batch)
            loss = criterion(logits, label_batch)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item() * audio_batch.size(0)
            
        epoch_loss = running_loss / len(train_dataset)
        
        # Test Evaluation at the end of each epoch
        model.eval()
        all_preds, all_trues = [], []
        with torch.no_grad():
            for audio_batch, label_batch in test_loader:
                audio_batch = audio_batch.to(device)
                logits = model(audio_batch)
                preds = torch.argmax(logits, dim=1).cpu().numpy()
                
                all_preds.extend(preds)
                all_trues.extend(label_batch.numpy())

        acc = accuracy_score(all_trues, all_preds)
        print(f"Epoch {epoch+1:02d} Complete | Loss: {epoch_loss:.4f} | Validation Acc: {acc * 100:.2f}%")
        
        if acc > best_acc:
            best_acc = acc
            torch.save(model.state_dict(), "unlocked_clap_best.pt")

    elapsed = time.time() - start_time
    hours, rem = divmod(elapsed, 3600)
    mins, secs = divmod(rem, 60)

    print(f"\n======================================")
    print(f"Unlocked Fine-Tuning Best Test Accuracy: {best_acc * 100:.2f}%")
    print(f"Total Time Elapsed: {int(hours)}h {int(mins)}m {secs:.2f}s")
    print(f"======================================")
    
    print("\nDetailed Language Classification Report:")
    target_names = [unique_langs[i] for i in range(num_classes)]
    print(classification_report(all_trues, all_preds, target_names=target_names))

if __name__ == "__main__":
    DATASET_PATH = "ekstep_seen"
    run_unlocked_finetune(DATASET_PATH, batch_size=32, epochs=10, lr=1e-4)