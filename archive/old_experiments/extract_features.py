# extract_features.py
import os
import glob
import torch
import torch.nn.functional as F
from tqdm import tqdm
from evaluator import load_clap_model
from preprocess import clean_and_format_audio

FOLDER_TO_LANG = {
    "hin": 0, "ben": 1, "mar": 2, "tel": 3, "tam": 4, "guj": 5, "urd": 6, 
    "kan": 7, "ori": 8, "odi": 8, "mal": 9, "pun": 10, "asm": 11, "mai": 12, 
    "sat": 13, "kas": 14, "nep": 15, "snd": 16, "kok": 17, "doi": 18, 
    "mni": 19, "brx": 20, "san": 21, "eng": 22
}

def get_audio_embedding(file_path, model, processor):
    """Bypasses text completely. Just extracts the pure 512D audio vector."""
    audio_chunks, sr = clean_and_format_audio(file_path)
    audio_inputs = processor(audio=audio_chunks, sampling_rate=sr, return_tensors="pt", padding=True)
    
    with torch.no_grad():
        outputs = model.get_audio_features(**audio_inputs)
        
        # Safely extract tensor
        if isinstance(outputs, torch.Tensor):
            audio_embeds = outputs
        elif hasattr(outputs, "audio_embeds") and outputs.audio_embeds is not None:
            audio_embeds = outputs.audio_embeds
        elif hasattr(outputs, "pooler_output"):
            audio_embeds = outputs.pooler_output
        else:
            audio_embeds = outputs[0]
            
        # Version-proof projection check
        if audio_embeds.shape[-1] != model.config.projection_dim:
            audio_embeds = model.audio_projection(audio_embeds)
            
    # Temporal Ensembling: Average the 3 chunks
    mean_audio_embed = torch.mean(audio_embeds, dim=0, keepdim=True)
    return F.normalize(mean_audio_embed, p=2, dim=-1)

def extract_and_save_features(dataset_root, output_filename):
    print(f"\nStarting extraction for: {dataset_root}")
    model, processor = load_clap_model()
    
    wav_files = glob.glob(os.path.join(dataset_root, "*", "*.wav"))
    if not wav_files:
        print(f"No .wav files found in {dataset_root}")
        return

    embeddings_list = []
    labels_list = []
    

    for file_path in tqdm(wav_files, desc="Extracting"):
        folder = os.path.basename(os.path.dirname(file_path))
        label_idx = FOLDER_TO_LANG.get(folder)
        
        if label_idx is None:
            continue
            
        try:
            embed = get_audio_embedding(file_path, model, processor)
            embeddings_list.append(embed.squeeze(0)) # Remove batch dimension
            labels_list.append(torch.tensor(label_idx))
        except Exception as e:
            print(f"\nError processing {file_path}: {e}")
            

    X = torch.stack(embeddings_list)
    y = torch.stack(labels_list)
    
    torch.save({'embeddings': X, 'labels': y}, output_filename)
    print(f"Successfully saved {len(y)} embeddings to {output_filename}")

if __name__ == "__main__":

    TRAIN_DIR = "ekstep_seen/Ekstep_12_lang_multi_domain" 
    extract_and_save_features(TRAIN_DIR, "train_features.pt")

    TEST_DIR = "IITMandi_YouTube"
    extract_and_save_features(TEST_DIR, "test_features.pt")