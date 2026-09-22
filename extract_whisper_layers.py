"""
Deep Dive #2: Layer-Wise Whisper Extraction
Extracts features from Whisper's Layer 2, Layer 4, and Layer 6 to see which 
level of abstraction serves as the best 'gatekeeper' against background noise.
"""
import torch
import soundfile as sf
import torchaudio.transforms as T
import os
import numpy as np
from tqdm import tqdm
from transformers import AutoFeatureExtractor, WhisperModel

# --- SETUP ---
FOLDER_TO_LANG = {
    "hin": "Hindi", "ben": "Bengali", "mar": "Marathi", "tel": "Telugu",
    "tam": "Tamil", "guj": "Gujarati", "urd": "Urdu", "kan": "Kannada",
    "ori": "Odia", "odi": "Odia", "mal": "Malayalam", "pun": "Punjabi",
    "asm": "Assamese", "eng": "English"
}
SEEN_LANGS = ['Marathi', 'Kannada', 'English', 'Odia', 'Punjabi', 'Malayalam', 'Telugu', 'Gujarati']
DEVICE = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')

def load_audio_for_whisper(path, target_sr=16000):
    audio, sr = sf.read(path)
    if len(audio.shape) > 1:
        audio = audio.mean(axis=1) # to mono
    if sr != target_sr:
        resampler = T.Resample(sr, target_sr)
        audio = resampler(torch.FloatTensor(audio)).numpy()
    return audio

def extract_whisper_layers(folder_path, langs, cache_name):
    print(f"\nInitializing Whisper to extract multi-layer features for {folder_path}...")
    
    whisper_id = "openai/whisper-base"
    processor = AutoFeatureExtractor.from_pretrained(whisper_id)
    whisper_model = WhisperModel.from_pretrained(whisper_id).encoder.to(DEVICE)
    whisper_model.eval()
    
    # EXACT SAME SORTING LOGIC TO GUARANTEE ALIGNMENT WITH CLAP FEATURES
    lang_to_idx = {l: i for i, l in enumerate(langs)}
    files_to_process = []
    
    for root, _, files in os.walk(folder_path):
        folder_name = os.path.basename(root)
        if folder_name in FOLDER_TO_LANG:
            lang_name = FOLDER_TO_LANG[folder_name]
            if lang_name in langs:
                for f in files:
                    if f.endswith('.wav'):
                        files_to_process.append((os.path.join(root, f), lang_to_idx[lang_name]))
    
    files_to_process.sort(key=lambda x: x[0])
    print(f"Found {len(files_to_process)} files. Extracting Layers 2, 4, and 6...")
    
    layer2_features = []
    layer4_features = []
    layer6_features = []
    labels = []
    
    for path, label in tqdm(files_to_process, desc="Extracting Layers"):
        try:
            w_audio = load_audio_for_whisper(path)
            inputs = processor(w_audio, sampling_rate=16000, return_tensors="pt")
            w_input = inputs.input_features.to(DEVICE)
            
            with torch.no_grad():
                # output_hidden_states=True returns a tuple of all layers
                out = whisper_model(w_input, output_hidden_states=True)
                hidden_states = out.hidden_states
                
                # Mean pool each layer across the time dimension
                # Index 0 is the embedding layer, Index 2 is Layer 2, etc.
                l2 = hidden_states[2].mean(dim=1).squeeze().cpu().numpy()
                l4 = hidden_states[4].mean(dim=1).squeeze().cpu().numpy()
                l6 = hidden_states[6].mean(dim=1).squeeze().cpu().numpy()
            
            layer2_features.append(l2)
            layer4_features.append(l4)
            layer6_features.append(l6)
            labels.append(label)
        except Exception as e:
            pass # Skip corrupted files (will align with CLAP skip logic)
            
    X_l2 = np.array(layer2_features)
    X_l4 = np.array(layer4_features)
    X_l6 = np.array(layer6_features)
    y = np.array(labels)
    
    np.savez(cache_name, X_layer2=X_l2, X_layer4=X_l4, X_layer6=X_l6, y=y)
    print(f"Saved multi-layer features to {cache_name}")

if __name__ == "__main__":
    extract_whisper_layers("ekstep_seen", SEEN_LANGS, "whisper_layers_ekstep.npz")
    extract_whisper_layers("IITMandi_YouTube", SEEN_LANGS, "whisper_layers_youtube.npz")
