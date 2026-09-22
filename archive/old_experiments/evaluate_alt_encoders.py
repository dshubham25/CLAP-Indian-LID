import torch
import torch.nn as nn
from transformers import AutoFeatureExtractor, AutoModel, WhisperModel, Wav2Vec2Model, WavLMModel, Wav2Vec2ConformerModel
import soundfile as sf
import torchaudio.transforms as T
import os
import numpy as np
from tqdm import tqdm
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
import argparse

# --- DATASET ---
FOLDER_TO_LANG = {
    "hin": "Hindi", "ben": "Bengali", "mar": "Marathi", "tel": "Telugu",
    "tam": "Tamil", "guj": "Gujarati", "urd": "Urdu", "kan": "Kannada",
    "ori": "Odia", "odi": "Odia", "mal": "Malayalam", "pun": "Punjabi",
    "asm": "Assamese", "eng": "English"
}
SEEN_LANGS = ['Marathi', 'Kannada', 'English', 'Odia', 'Punjabi', 'Malayalam', 'Telugu', 'Gujarati']
# Unseen languages cannot be evaluated with a Linear Probe trained on seen languages, 
# because audio-only models lack the text-embedding space required for zero-shot transfer!
UNSEEN_LANGS = ['Hindi', 'Assamese', 'Bengali', 'Tamil']

MODEL_DICT = {
    "whisper": "openai/whisper-base",
    "wavlm": "microsoft/wavlm-base-plus",
    "wav2vec2": "facebook/wav2vec2-base",
    "conformer": "facebook/wav2vec2-conformer-rel-pos-large",
    "distilhubert": "ntu-spml/distilhubert"
}

def load_audio(path, target_sr=16000):
    audio, sr = sf.read(path)
    if len(audio.shape) > 1:
        audio = audio.mean(axis=1) # to mono
    if sr != target_sr:
        resampler = T.Resample(sr, target_sr)
        audio = resampler(torch.FloatTensor(audio)).numpy()
    return audio

def extract_features(model_name, folder_path, langs):
    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    print(f"Loading {model_name} on {device}...")
    
    model_id = MODEL_DICT[model_name]
    processor = AutoFeatureExtractor.from_pretrained(model_id)
    
    if "whisper" in model_name:
        model = WhisperModel.from_pretrained(model_id).encoder
    elif "wavlm" in model_name:
        model = WavLMModel.from_pretrained(model_id)
    elif "conformer" in model_name:
        model = Wav2Vec2ConformerModel.from_pretrained(model_id)
    elif "distilhubert" in model_name:
        # DistilHuBERT uses the same architecture as Wav2Vec2 base internally
        from transformers import HubertModel
        model = HubertModel.from_pretrained(model_id)
    else:
        model = Wav2Vec2Model.from_pretrained(model_id)
        
    model = model.to(device)
    model.eval()
    
    features = []
    labels = []
    lang_to_idx = {l: i for i, l in enumerate(langs)}
    
    # Gather files
    files_to_process = []
    for root, _, files in os.walk(folder_path):
        folder_name = os.path.basename(root)
        if folder_name in FOLDER_TO_LANG:
            lang_name = FOLDER_TO_LANG[folder_name]
            if lang_name in langs:
                for f in files:
                    if f.endswith('.wav'):
                        files_to_process.append((os.path.join(root, f), lang_to_idx[lang_name]))
                        
    print(f"Found {len(files_to_process)} files for {len(langs)} languages.")
    
    # We will process a small subset for demonstration if the dataset is huge
    # In a real run, this loop processes everything
    for path, label in tqdm(files_to_process, desc="Extracting features"):
        audio = load_audio(path)
        
        # Whisper expects different processing than wav2vec family
        if "whisper" in model_name:
            inputs = processor(audio, sampling_rate=16000, return_tensors="pt")
            input_features = inputs.input_features.to(device)
            with torch.no_grad():
                out = model(input_features)
                # Mean pooling over the sequence dimension
                emb = out.last_hidden_state.mean(dim=1).squeeze().cpu().numpy()
        else:
            inputs = processor(audio, sampling_rate=16000, return_tensors="pt")
            input_values = inputs.input_values.to(device)
            with torch.no_grad():
                out = model(input_values)
                emb = out.last_hidden_state.mean(dim=1).squeeze().cpu().numpy()
                
        features.append(emb)
        labels.append(label)
        
    return np.array(features), np.array(labels)

def train_probe(X_train, y_train, X_test, y_test):
    print("Training Logistic Regression probe...")
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    clf = LogisticRegression(max_iter=1000)
    clf.fit(X_train, y_train)
    
    preds = clf.predict(X_test)
    acc = accuracy_score(y_test, preds)
    print(f"Supervised Classification Accuracy: {acc*100:.2f}%")
    return acc

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, choices=list(MODEL_DICT.keys()), required=True)
    parser.add_argument("--subset", type=int, default=0, help="Number of samples to run (0 for all)")
    parser.add_argument("--test-dataset", type=str, default=None, help="Dataset to test on (e.g., IITMandi_YouTube)")
    args = parser.parse_args()
    
    print("WARNING: Audio-only models (WaveLM, Wav2Vec2) lack a paired text encoder.")
    print("They cannot perform Zero-Shot classification on completely unseen languages.")
    print("This script trains a supervised linear probe to evaluate their representation power.")
    
    cache_file = f"features_{args.model}.npz"
    if os.path.exists(cache_file) and args.subset == 0:
        print(f"Found cached train features at {cache_file}. Loading from disk...")
        data = np.load(cache_file)
        X, y = data['X'], data['y']
    else:
        X, y = extract_features(args.model, "ekstep_seen", SEEN_LANGS)
        if args.subset == 0:
            np.savez(cache_file, X=X, y=y)
            print(f"Saved extracted train features to {cache_file}")
            
    if args.test_dataset:
        test_cache = f"features_{args.model}_{args.test_dataset}.npz"
        if os.path.exists(test_cache) and args.subset == 0:
            print(f"Found cached test features at {test_cache}. Loading from disk...")
            data = np.load(test_cache)
            X_test, y_test = data['X'], data['y']
        else:
            X_test, y_test = extract_features(args.model, args.test_dataset, SEEN_LANGS)
            if args.subset == 0:
                np.savez(test_cache, X=X_test, y=y_test)
                print(f"Saved extracted test features to {test_cache}")
        
        X_train, y_train = X, y
    else:
        if args.subset > 0:
            X = X[:args.subset]
            y = y[:args.subset]
        
        # Simple train/test split for demonstration
        from sklearn.model_selection import train_test_split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    train_probe(X_train, y_train, X_test, y_test)
