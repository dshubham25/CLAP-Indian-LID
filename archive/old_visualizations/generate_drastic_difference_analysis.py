import os
import glob
import types
import torch
import soundfile as sf
import torchaudio.transforms as T
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
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

def estimate_snr(audio_path):
    try:
        audio_array, _ = sf.read(audio_path)
        if len(audio_array.shape) > 1:
            audio_array = audio_array.mean(axis=1) # to mono
        frame_length = 512
        num_frames = len(audio_array) // frame_length
        if num_frames == 0: return 0
        audio_array = audio_array[:num_frames * frame_length]
        frames = audio_array.reshape(num_frames, frame_length)
        powers = np.sum(frames**2, axis=1) / frame_length
        powers = powers[powers > 0]
        if len(powers) < 10: return 0
        powers_sorted = np.sort(powers)
        noise_power = np.mean(powers_sorted[:max(1, len(powers)//10)])
        signal_power = np.mean(powers_sorted[-max(1, len(powers)//10):])
        if noise_power == 0: return 40.0
        return 10 * np.log10(signal_power / noise_power)
    except Exception as e:
        return 0

def run_analysis():
    print("Initializing MS-CLAP...")
    clap_model = CLAP(version='2023', use_cuda=False)
    clap_model.read_audio = types.MethodType(custom_read_audio, clap_model)

    prompts = [f"A person speaking in {lang}" for lang in SEEN_LANGS]
    print("Extracting Text Prompt Embeddings...")
    text_embeddings = clap_model.get_text_embeddings(prompts).detach()

    ekstep_files = glob.glob(os.path.join("ekstep_seen", "**", "*.wav"), recursive=True)
    iitmandi_files = glob.glob(os.path.join("IITMandi_YouTube", "**", "*.wav"), recursive=True)
    
    np.random.seed(42)
    np.random.shuffle(ekstep_files)
    np.random.shuffle(iitmandi_files)
    
    # Take 200 files from each for quick analysis
    ekstep_subset = [f for f in ekstep_files if os.path.basename(os.path.dirname(f)) in FOLDER_TO_LANG and FOLDER_TO_LANG[os.path.basename(os.path.dirname(f))] in SEEN_LANGS][:200]
    iitmandi_subset = [f for f in iitmandi_files if os.path.basename(os.path.dirname(f)) in FOLDER_TO_LANG and FOLDER_TO_LANG[os.path.basename(os.path.dirname(f))] in SEEN_LANGS][:200]

    results = {
        "ekstep_snr": [], "iitmandi_snr": [],
        "ekstep_sim": [], "iitmandi_sim": [],
        "iitmandi_true": [], "iitmandi_pred": []
    }

    print("Analyzing Ekstep (In-Domain)...")
    for f in ekstep_subset:
        true_lang = FOLDER_TO_LANG[os.path.basename(os.path.dirname(f))]
        true_idx = SEEN_LANGS.index(true_lang)
        
        results["ekstep_snr"].append(estimate_snr(f))
        try:
            emb = clap_model.get_audio_embeddings([f], resample=True).detach()
            sims = clap_model.compute_similarity(emb, text_embeddings).squeeze(0).detach().cpu().numpy()
            results["ekstep_sim"].append(sims[true_idx])
        except Exception as e:
            print("Ekstep Error:", e)

    print("Analyzing IITMandi (Cross-Domain)...")
    for f in iitmandi_subset:
        true_lang = FOLDER_TO_LANG[os.path.basename(os.path.dirname(f))]
        true_idx = SEEN_LANGS.index(true_lang)
        
        results["iitmandi_snr"].append(estimate_snr(f))
        try:
            emb = clap_model.get_audio_embeddings([f], resample=True).detach()
            sims = clap_model.compute_similarity(emb, text_embeddings).squeeze(0).detach().cpu().numpy()
            results["iitmandi_sim"].append(sims[true_idx])
            pred_idx = np.argmax(sims)
            results["iitmandi_true"].append(true_lang)
            results["iitmandi_pred"].append(SEEN_LANGS[pred_idx])
        except Exception as e:
            print("IITMandi Error:", e)

    print("Generating Plots...")
    
    # 1. SNR Plot
    plt.figure(figsize=(8, 5))
    sns.kdeplot(results["ekstep_snr"], fill=True, label="Ekstep (In-Domain)", color="blue")
    sns.kdeplot(results["iitmandi_snr"], fill=True, label="IITMandi (Cross-Domain)", color="red")
    plt.title("Signal-to-Noise Ratio (SNR) Distribution")
    plt.xlabel("SNR (dB) - Higher is cleaner")
    plt.ylabel("Density")
    plt.legend()
    plt.savefig("analysis_snr.png", bbox_inches='tight')
    plt.close()

    # 2. Similarity Plot
    plt.figure(figsize=(8, 5))
    sns.kdeplot(results["ekstep_sim"], fill=True, label="Ekstep (In-Domain)", color="blue")
    sns.kdeplot(results["iitmandi_sim"], fill=True, label="IITMandi (Cross-Domain)", color="red")
    plt.title("Cosine Similarity for CORRECT Language Prompt")
    plt.xlabel("Similarity Score")
    plt.ylabel("Density")
    plt.legend()
    plt.savefig("analysis_similarity.png", bbox_inches='tight')
    plt.close()

    # 3. Confusion Matrix
    cm = confusion_matrix(results["iitmandi_true"], results["iitmandi_pred"], labels=SEEN_LANGS)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Reds', xticklabels=SEEN_LANGS, yticklabels=SEEN_LANGS)
    plt.title("Cross-Domain Zero-Shot Confusion Matrix")
    plt.xlabel("Predicted Language")
    plt.ylabel("True Language")
    plt.xticks(rotation=45)
    plt.savefig("analysis_confusion.png", bbox_inches='tight')
    plt.close()

    print("Done! Saved analysis_snr.png, analysis_similarity.png, analysis_confusion.png")

if __name__ == "__main__":
    run_analysis()
