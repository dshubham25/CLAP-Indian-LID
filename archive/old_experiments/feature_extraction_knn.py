import os
import glob
import time
import types
import torch
import soundfile as sf
import torchaudio.transforms as T
import numpy as np
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report
from msclap import CLAP

FOLDER_TO_LANG = {
    "hin": "Hindi", "ben": "Bengali", "mar": "Marathi", "tel": "Telugu",
    "tam": "Tamil", "guj": "Gujarati", "urd": "Urdu", "kan": "Kannada",
    "ori": "Odia", "odi": "Odia", "mal": "Malayalam", "pun": "Punjabi",
    "asm": "Assamese", "mai": "Maithili", "sat": "Santali", "kas": "Kashmiri",
    "nep": "Nepali", "snd": "Sindhi", "kok": "Konkani", "doi": "Dogri",
    "mni": "Manipuri", "brx": "Bodo", "san": "Sanskrit", "eng": "English",
}

# --- THE MONKEY PATCH (Mac Fix) ---
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

def run_knn_stratified_split(dataset_root, features_file="clap_features.npz"):
    # If features already exist, skip the 1.5 hour extraction and load them instantly
    if os.path.exists(features_file):
        print(f"Found saved features at {features_file}. Loading into memory...")
        data = np.load(features_file)
        X = data['X']
        y = data['y']
        print(f"Loaded {len(X)} samples successfully.")
    else:
        print("No saved features found. Initializing MS-CLAP for extraction...")
        clap_model = CLAP(version='2023', use_cuda=False)
        clap_model.read_audio = types.MethodType(custom_read_audio, clap_model)
        
        wav_files = glob.glob(os.path.join(dataset_root, "**", "*.wav"), recursive=True)
        if not wav_files:
            print(f"No .wav files found in {dataset_root}.")
            return
            
        embeddings = []
        labels = []
        
        print(f"Extracting features for {len(wav_files)} files. This will take some time...")
        for file_path in tqdm(wav_files, desc="Extracting"):
            folder = os.path.basename(os.path.dirname(file_path))
            if folder not in FOLDER_TO_LANG:
                continue
                
            try:
                # Extract the 512D audio vector
                audio_tensor = clap_model.get_audio_embeddings([file_path], resample=True)
                embeddings.append(audio_tensor.squeeze().detach().cpu().numpy())
                labels.append(FOLDER_TO_LANG[folder])
            except Exception as e:
                # Skip corrupted files silently to keep the loop moving
                pass
                
        X = np.array(embeddings)
        y = np.array(labels)
        
        # Save features so you never have to extract them again
        print(f"\nSaving features to {features_file}...")
        np.savez(features_file, X=X, y=y)

    print("\n--- Applying 70/30 Stratified File Split ---")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
    
    print(f"Training KNN Classifier on {len(X_train)} samples...")
    knn = KNeighborsClassifier(n_neighbors=5, metric='cosine')
    knn.fit(X_train, y_train)
    
    print(f"Testing KNN Classifier on {len(X_test)} unseen samples...")
    y_pred = knn.predict(X_test)
    
    acc = accuracy_score(y_test, y_pred)
    print(f"\n======================================")
    print(f"KNN Classification Accuracy: {acc * 100:.2f}%")
    print(f"======================================")
    print("\nDetailed Language Breakdown:\n")
    print(classification_report(y_test, y_pred))

if __name__ == "__main__":
    DATASET_PATH = "ekstep_seen"
    run_knn_stratified_split(DATASET_PATH)