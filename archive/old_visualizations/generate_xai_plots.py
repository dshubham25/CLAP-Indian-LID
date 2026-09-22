import os
import glob
import types
import torch
import soundfile as sf
import torchaudio.transforms as T
import numpy as np
import matplotlib.pyplot as plt
import shap
from msclap import CLAP

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

def run_shap_and_plot_lime():
    print("Initializing MS-CLAP for SHAP...")
    clap_model = CLAP(version='2023', use_cuda=False)
    clap_model.read_audio = types.MethodType(custom_read_audio, clap_model)

    audio_files = glob.glob(os.path.join("ekstep_seen", "**", "*.wav"), recursive=True)
    audio_files = [f for f in audio_files if "/mar/" in f or "/marathi/" in f.lower()]
    test_audio = audio_files[0]
    audio_emb = clap_model.get_audio_embeddings([test_audio], resample=True).detach()
    
    def predictor(texts):
        # SHAP might pass numpy arrays of texts, so convert to list
        if isinstance(texts, np.ndarray):
            texts = texts.tolist()
        
        # If texts is empty for some reason
        if len(texts) == 0:
            return np.array([])
            
        text_embs = clap_model.get_text_embeddings(texts)
        sims = clap_model.compute_similarity(audio_emb, text_embs).squeeze(0).detach().cpu().numpy() 
        probs = 1 / (1 + np.exp(-sims)) 
        results = np.zeros((len(texts), 2))
        results[:, 1] = probs
        results[:, 0] = 1 - probs
        return results

    text_to_explain = "A person speaking in Marathi"
    print(f"\nRunning SHAP Explainer on: '{text_to_explain}'")
    
    # SHAP Text Masker
    masker = shap.maskers.Text(r"\W")
    explainer = shap.Explainer(predictor, masker, output_names=["Mismatch", "Match"])
    shap_values = explainer([text_to_explain])
    
    # SHAP Plotting
    plt.figure(figsize=(10, 4))
    # We plot the SHAP values for the "Match" class (index 1)
    shap.plots.waterfall(shap_values[0, :, 1], show=False)
    plt.tight_layout()
    plt.savefig("shap_plot.png", bbox_inches='tight', dpi=300)
    plt.close()
    print("Saved SHAP waterfall plot to shap_plot.png")

    # LIME Plotting (using the values we already calculated)
    print("\nGenerating LIME Bar Chart...")
    lime_features = ['Marathi', 'speaking', 'A', 'in', 'person']
    lime_weights = [0.0044, 0.0007, 0.0001, 0.0000, -0.0015]
    
    plt.figure(figsize=(10, 4))
    colors = ['green' if w > 0 else 'red' for w in lime_weights]
    plt.barh(lime_features, lime_weights, color=colors)
    plt.xlabel('LIME Importance Weight')
    plt.title('LIME Feature Importance for Zero-Shot Prompt')
    plt.axvline(0, color='black', linewidth=1)
    
    for i, v in enumerate(lime_weights):
        plt.text(v + (0.0001 if v > 0 else -0.0005), i, f"{v:.4f}", color='black', va='center')
        
    plt.tight_layout()
    plt.savefig("lime_plot.png", bbox_inches='tight', dpi=300)
    plt.close()
    print("Saved LIME bar chart to lime_plot.png")

if __name__ == "__main__":
    run_shap_and_plot_lime()
