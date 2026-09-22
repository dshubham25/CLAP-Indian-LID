import os
import glob
import time
import types
import torch
import soundfile as sf
import torchaudio.transforms as T
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import seaborn as sns
from msclap import CLAP

FOLDER_TO_LANG = {
    "hin": "Hindi", "ben": "Bengali", "mar": "Marathi", "tel": "Telugu",
    "tam": "Tamil", "guj": "Gujarati", "urd": "Urdu", "kan": "Kannada",
    "ori": "Odia", "odi": "Odia", "mal": "Malayalam", "pun": "Punjabi",
    "eng": "English"
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

def run_tsne():
    print("Initializing MS-CLAP...")
    clap_model = CLAP(version='2023', use_cuda=False)
    clap_model.read_audio = types.MethodType(custom_read_audio, clap_model)

    # 1. Get Audio Embeddings for in-domain (ekstep)
    ekstep_files = glob.glob(os.path.join("ekstep_seen", "**", "*.wav"), recursive=True)
    iitmandi_files = glob.glob(os.path.join("IITMandi_YouTube", "**", "*.wav"), recursive=True)
    
    np.random.seed(42)
    np.random.shuffle(ekstep_files)
    np.random.shuffle(iitmandi_files)
    
    ekstep_subset = [f for f in ekstep_files if os.path.basename(os.path.dirname(f)) in FOLDER_TO_LANG and FOLDER_TO_LANG[os.path.basename(os.path.dirname(f))] in SEEN_LANGS][:200]
    iitmandi_subset = [f for f in iitmandi_files if os.path.basename(os.path.dirname(f)) in FOLDER_TO_LANG and FOLDER_TO_LANG[os.path.basename(os.path.dirname(f))] in SEEN_LANGS][:200]

    all_embeddings = []
    labels = []

    print("Extracting Ekstep Audio Embeddings (In-Domain)...")
    for f in ekstep_subset:
        try:
            emb = clap_model.get_audio_embeddings([f], resample=True)
            all_embeddings.append(emb.detach().cpu().numpy())
            labels.append("Ekstep (In-Domain Audio)")
        except:
            pass

    print("Extracting IITMandi Audio Embeddings (Cross-Domain)...")
    for f in iitmandi_subset:
        try:
            emb = clap_model.get_audio_embeddings([f], resample=True)
            all_embeddings.append(emb.detach().cpu().numpy())
            labels.append("IITMandi (Cross-Domain Audio)")
        except:
            pass

    print("Extracting Text Prompt Embeddings...")
    prompts = [f"A person speaking in {lang}" for lang in SEEN_LANGS]
    text_embeddings = clap_model.get_text_embeddings(prompts).detach().cpu().numpy()
    
    for i, p in enumerate(prompts):
        all_embeddings.append(text_embeddings[i:i+1])
        labels.append("Text Prompts")
        
    X = np.vstack(all_embeddings)
    
    print("Running t-SNE...")
    tsne = TSNE(n_components=2, perplexity=30, random_state=42)
    X_tsne = tsne.fit_transform(X)
    
    plt.figure(figsize=(10, 8))
    sns.scatterplot(
        x=X_tsne[:, 0], y=X_tsne[:, 1],
        hue=labels, style=labels,
        palette={"Ekstep (In-Domain Audio)": "blue", "IITMandi (Cross-Domain Audio)": "red", "Text Prompts": "black"},
        s=100, alpha=0.7
    )
    
    # Highlight Text Prompts
    text_idx = [i for i, l in enumerate(labels) if l == "Text Prompts"]
    for i, idx in enumerate(text_idx):
        plt.annotate(SEEN_LANGS[i], (X_tsne[idx, 0], X_tsne[idx, 1]), 
                     fontsize=10, fontweight='bold',
                     bbox=dict(facecolor='white', alpha=0.7, edgecolor='none'))

    plt.title("t-SNE: Text Prompts vs In-Domain vs Cross-Domain Audio")
    plt.savefig("tsne_domain_shift.png")
    print("Saved t-SNE plot to tsne_domain_shift.png!")

if __name__ == "__main__":
    run_tsne()
