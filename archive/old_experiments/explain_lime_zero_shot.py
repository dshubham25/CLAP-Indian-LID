import os
import glob
import types
import torch
import soundfile as sf
import torchaudio.transforms as T
import numpy as np
from msclap import CLAP
from lime.lime_text import LimeTextExplainer

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

def run_lime():
    print("Initializing MS-CLAP...")
    clap_model = CLAP(version='2023', use_cuda=False)
    clap_model.read_audio = types.MethodType(custom_read_audio, clap_model)

    # Take a sample audio file from ekstep
    all_wavs = glob.glob(os.path.join("ekstep_seen", "**", "*.wav"), recursive=True)
    audio_files = [f for f in all_wavs if "/mar/" in f or "/marathi/" in f.lower()]
    if not audio_files:
        print("Could not find an audio file in ekstep_seen for marathi.")
        return
    
    test_audio = audio_files[0]
    print(f"Using audio file: {test_audio}")
    
    audio_emb = clap_model.get_audio_embeddings([test_audio], resample=True).detach()
    
    def predictor(texts):
        # Extract embeddings for the perturbed texts
        text_embs = clap_model.get_text_embeddings(texts)
        # Compute cosine similarity between all perturbed texts and the single audio clip
        sims = clap_model.compute_similarity(audio_emb, text_embs)
        sims = sims.squeeze(0).detach().cpu().numpy() 
        
        # Convert raw cosine similarity to a 0-1 probability score for LIME
        probs = 1 / (1 + np.exp(-sims)) 
        
        results = np.zeros((len(texts), 2))
        results[:, 1] = probs
        results[:, 0] = 1 - probs
        return results

    explainer = LimeTextExplainer(class_names=['Mismatch', 'Match'])
    
    text_to_explain = "A person speaking in Marathi"
    print(f"\nExplaining Zero-Shot Prompt: '{text_to_explain}'")
    
    # Run LIME with a smaller number of samples so it finishes quickly
    exp = explainer.explain_instance(text_to_explain, predictor, num_features=5, num_samples=100)
    
    print("\nLIME Feature Importance for Zero-Shot Text Prompt:")
    for feature, weight in exp.as_list():
        print(f"Word: '{feature:15s}' | Importance Weight: {weight:.4f}")
        
    exp.save_to_file("lime_explanation.html")
    print("\nSaved interactive LIME explanation to lime_explanation.html")

if __name__ == "__main__":
    run_lime()
