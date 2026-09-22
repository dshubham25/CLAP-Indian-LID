"""
Phase 1.3: CLAP Linear Probe — Fair comparison against Whisper Linear Probe.

Extracts CLAP audio embeddings (without any text/zero-shot), trains an identical
Logistic Regression linear probe, and evaluates on both in-domain and cross-domain.

This proves whether CLAP's audio features encode phonetic content or just acoustic texture.

Usage:
  # Step 1: Extract features & evaluate in-domain (80/20 split on ekstep_seen)
  python clap_linear_probe.py

  # Step 2: Evaluate cross-domain (train on ekstep_seen, test on IITMandi_YouTube)
  python clap_linear_probe.py --test-dataset IITMandi_YouTube
"""
import torch
import types
import soundfile as sf
import torchaudio.transforms as T
import os
import numpy as np
from tqdm import tqdm
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from msclap import CLAP
import argparse

FOLDER_TO_LANG = {
    "hin": "Hindi", "ben": "Bengali", "mar": "Marathi", "tel": "Telugu",
    "tam": "Tamil", "guj": "Gujarati", "urd": "Urdu", "kan": "Kannada",
    "ori": "Odia", "odi": "Odia", "mal": "Malayalam", "pun": "Punjabi",
    "asm": "Assamese", "eng": "English"
}
SEEN_LANGS = ['Marathi', 'Kannada', 'English', 'Odia', 'Punjabi', 'Malayalam', 'Telugu', 'Gujarati']

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

def extract_clap_features(folder_path, langs):
    print("Initializing MS-CLAP...")
    clap_model = CLAP(version='2023', use_cuda=False)
    clap_model.read_audio = types.MethodType(custom_read_audio, clap_model)
    
    features = []
    labels = []
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
    
    print(f"Found {len(files_to_process)} files for {len(langs)} languages.")
    
    for path, label in tqdm(files_to_process, desc="Extracting CLAP audio features"):
        try:
            emb = clap_model.get_audio_embeddings([path], resample=True).detach().cpu().numpy().squeeze()
            features.append(emb)
            labels.append(label)
        except Exception as e:
            pass
    
    return np.array(features), np.array(labels)

def train_probe(X_train, y_train, X_test, y_test, langs):
    print("Training Logistic Regression probe...")
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)
    
    clf = LogisticRegression(max_iter=1000)
    clf.fit(X_train_s, y_train)
    
    preds = clf.predict(X_test_s)
    acc = accuracy_score(y_test, preds)
    print(f"\n{'='*50}")
    print(f"CLAP LINEAR PROBE ACCURACY: {acc*100:.2f}%")
    print(f"{'='*50}\n")
    
    present_classes = sorted(list(set(y_test)))
    target_names = [langs[i] for i in present_classes]
    print(classification_report(y_test, preds, labels=present_classes, target_names=target_names, zero_division=0))
    return acc

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--test-dataset", type=str, default=None,
                        help="Cross-domain test dataset folder (e.g., IITMandi_YouTube)")
    args = parser.parse_args()
    
    # --- Extract or load cached train features ---
    cache_file = "features_clap_audio.npz"
    if os.path.exists(cache_file):
        print(f"Found cached train features at {cache_file}. Loading...")
        data = np.load(cache_file)
        X, y = data['X'], data['y']
    else:
        X, y = extract_clap_features("ekstep_seen", SEEN_LANGS)
        np.savez(cache_file, X=X, y=y)
        print(f"Saved train features to {cache_file}")
    
    if args.test_dataset:
        # --- Cross-domain evaluation ---
        test_cache = f"features_clap_audio_{args.test_dataset}.npz"
        if os.path.exists(test_cache):
            print(f"Found cached test features at {test_cache}. Loading...")
            data = np.load(test_cache)
            X_test, y_test = data['X'], data['y']
        else:
            X_test, y_test = extract_clap_features(args.test_dataset, SEEN_LANGS)
            np.savez(test_cache, X=X_test, y=y_test)
            print(f"Saved test features to {test_cache}")
        
        print(f"\nTrain: ekstep_seen ({len(X)} samples)")
        print(f"Test:  {args.test_dataset} ({len(X_test)} samples)")
        train_probe(X, y, X_test, y_test, SEEN_LANGS)
    else:
        # --- In-domain evaluation (80/20 split) ---
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        print(f"\nTrain: {len(X_train)} samples, Test: {len(X_test)} samples")
        train_probe(X_train, y_train, X_test, y_test, SEEN_LANGS)
