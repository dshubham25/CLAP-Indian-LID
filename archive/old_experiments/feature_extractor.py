import os
import glob
import numpy as np
import torch
import torchaudio
import librosa
from msclap import CLAP
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report
import torch.nn.functional as F

# --- MAC ENVIRONMENT BYPASS (Safety Catch) ---
import sys
sys.modules['torchcodec'] = None
os.environ["TORCHAUDIO_USE_TORCHCODEC"] = "0"

def safe_audio_load(filepath, *args, **kwargs):
    data, sr = librosa.load(filepath, sr=None, mono=False)
    tensor = torch.tensor(data, dtype=torch.float32)
    if tensor.ndim == 1:
        tensor = tensor.unsqueeze(0)
    return tensor, sr

torchaudio.load = safe_audio_load

# --- CONFIGURATION ---
DATASET_DIR = "./ekstep_seen" # Ensure this points to the main folder

# Comprehensive map to catch 3-letter codes AND full names
LANG_MAP = {
    "asm": "Assamese", "ben": "Bengali", "bho": "Bhojpuri", "dog": "Dogri",
    "guj": "Gujarati", "hin": "Hindi", "kan": "Kannada", "kas": "Kashmiri",
    "kok": "Konkani", "mai": "Maithili", "mal": "Malayalam", "mar": "Marathi",
    "nep": "Nepali", "odi": "Odia", "pan": "Punjabi", "san": "Santali",
    "snd": "Sindhi", "tam": "Tamil", "tel": "Telugu", "urd": "Urdu",
    "bhi": "Bhili", "gon": "Gondi",
    # Full names in lowercase
    "assamese": "Assamese", "bengali": "Bengali", "gujarati": "Gujarati",
    "hindi": "Hindi", "kannada": "Kannada", "malayalam": "Malayalam",
    "marathi": "Marathi", "odia": "Odia", "punjabi": "Punjabi",
    "tamil": "Tamil", "telugu": "Telugu"
}

# Extract unique languages from the map dynamically
LANGUAGES = sorted(list(set(LANG_MAP.values())))

# Create a mapping from language to an integer ID for the classifier
LANG_TO_ID = {lang: idx for idx, lang in enumerate(LANGUAGES)}
ID_TO_LANG = {idx: lang for lang, idx in LANG_TO_ID.items()}
# --- 1. SETUP MODEL & TEXT PROMPTS ---
use_gpu = torch.cuda.is_available()
device = torch.device('cuda' if use_gpu else 'cpu')
print(f"Loading MS-CLAP on {device}...")

clap_model = CLAP(version='2023', use_cuda=use_gpu)

# Define the text prompts for all languages as requested
text_prompts = [f"A person speaking in the {lang} language." for lang in LANGUAGES]

print("Extracting Text Embeddings for prompts...")
with torch.no_grad():
    text_embeddings = clap_model.get_text_embeddings(text_prompts).to(device)

# --- 2. FEATURE EXTRACTION ---
print(f"Scanning {DATASET_DIR} for audio files...")
audio_files = glob.glob(os.path.join(DATASET_DIR, "**", "*.wav"), recursive=True)

if not audio_files:
    print("ERROR: No .wav files found. Check your dataset path.")
    exit()

X_features = []
y_labels = []

print("Extracting Audio Features and Prompt Similarities...")
with torch.no_grad():
    for filepath in audio_files:
        matched_lang = None
        # Extract language name from folder structure (Assumes folder name is the language)
        path_parts = filepath.lower().replace('\\', '/').split('/')
        for part in path_parts:
            if part in LANG_MAP: 
                matched_lang = LANG_MAP[part]; 
                break
        
        # # Match folder name to our language list safely
        # matched_lang = None
        # for lang in LANGUAGES:
        #     if lang.lower() in folder_name.lower():
        #         matched_lang = lang
        #         break
                
        if not matched_lang:
            continue # Skip files that don't match our language list
            
        try:
            # 1. Get 1024-D Audio Embedding
            audio_emb = clap_model.get_audio_embeddings([filepath]).to(device)
            
            # 2. Get similarity scores between this audio and ALL text prompts
            # This incorporates your prompt requirement directly into the classifier's brain
            prompt_similarities = F.cosine_similarity(audio_emb, text_embeddings)
            
            # 3. Fuse them: 1024 Audio Features + N Prompt Similarities
            fused_features = torch.cat((audio_emb.flatten(), prompt_similarities.flatten()))
            
            # Save to our dataset lists
            X_features.append(fused_features.cpu().numpy())
            y_labels.append(LANG_TO_ID[matched_lang])
            
        except Exception as e:
            print(f"Failed to process {filepath}: {e}")

# Convert to standard Numpy arrays for Scikit-Learn
X = np.array(X_features)
y = np.array(y_labels)

print(f"Successfully extracted features for {len(X)} files. Feature vector size: {X.shape[1]}")

# --- 3. TRAIN / TEST SPLIT ---
# 80% of ekstep goes to training the classifier, 20% is held back for unseen testing
print("Splitting dataset into Train and Test sets...")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# --- 4. BUILD & TRAIN THE CLASSIFIER ---
print("Training K-Nearest Neighbors (KNN) Classifier...")
# We use metric='cosine' because MS-CLAP embeddings are optimized for Cosine Similarity during contrastive training.
# n_neighbors=5 is a standard benchmark for robust feature clustering evaluation.
classifier = KNeighborsClassifier(n_neighbors=5, metric='cosine')
classifier.fit(X_train, y_train)

# --- 5. EVALUATION ---
print("Evaluating on Test Set...")
y_pred = classifier.predict(X_test)

accuracy = accuracy_score(y_test, y_pred) * 100

print("\n" + "="*50)
print("EKSTEP IN-DOMAIN CLASSIFIER RESULTS")
print("="*50)
print(f"Training Samples : {len(X_train)}")
print(f"Testing Samples  : {len(X_test)}")
print(f"Final Accuracy   : {accuracy:.2f}%")
print("="*50)

# Print a detailed breakdown per languag
# Dynamically find which language IDs actually appear in the test set/predictions
unique_labels = np.unique(np.concatenate((y_test, y_pred)))
target_names = [ID_TO_LANG[i] for i in unique_labels]

print(classification_report(y_test, y_pred, labels=unique_labels, target_names=target_names))