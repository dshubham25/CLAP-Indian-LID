# evaluator.py
import torch
import torch.nn.functional as F
from transformers import ClapModel, ClapProcessor
from preprocess import clean_and_format_audio
from prompts import EKSTEP_PROMPTS

def load_clap_model():
    print("Loading Hugging Face CLAP model...")
    model = ClapModel.from_pretrained("laion/clap-htsat-unfused")
    processor = ClapProcessor.from_pretrained("laion/clap-htsat-unfused")
    model.eval()
    return model, processor

def get_averaged_text_embeddings(model, processor, prompts):
    inputs = processor(text=prompts, return_tensors="pt", padding=True)
    with torch.no_grad():
        outputs = model.get_text_features(**inputs)
        
        if isinstance(outputs, torch.Tensor):
            text_embeds = outputs
        elif hasattr(outputs, "text_embeds") and outputs.text_embeds is not None:
            text_embeds = outputs.text_embeds
        elif hasattr(outputs, "pooler_output"):
            text_embeds = outputs.pooler_output
        else:
            text_embeds = outputs[0]
            
        if text_embeds.shape[-1] != model.config.projection_dim:
            text_embeds = model.text_projection(text_embeds)

    mean_embedding = torch.mean(text_embeds, dim=0, keepdim=True)
    return F.normalize(mean_embedding, p=2, dim=-1)

def predict_language(audio_path, model, processor):
    audio_chunks, sr = clean_and_format_audio(audio_path)
    
    # Process all 3 to 9 augmented chunks simultaneously
    audio_inputs = processor(audio=audio_chunks, sampling_rate=sr, return_tensors="pt", padding=True)
    
    with torch.no_grad():
        outputs = model.get_audio_features(**audio_inputs)
        
        if isinstance(outputs, torch.Tensor):
            audio_embeds = outputs
        elif hasattr(outputs, "audio_embeds") and outputs.audio_embeds is not None:
            audio_embeds = outputs.audio_embeds
        elif hasattr(outputs, "pooler_output"):
            audio_embeds = outputs.pooler_output
        else:
            audio_embeds = outputs[0]
            
        if audio_embeds.shape[-1] != model.config.projection_dim:
            audio_embeds = model.audio_projection(audio_embeds)
            
        # Average the augmented chunks into one highly robust Super Embedding
        mean_audio_embed = torch.mean(audio_embeds, dim=0, keepdim=True)
        
        # ==========================================
        # TECHNIQUE 2: LATENT NOISE SUBTRACTION
        # ==========================================
        # 1. Define what noise sounds like in text
        noise_prompts = [
            "Loud background music, heavy traffic noise, static, and ambient sounds without speech.",
            "Environmental noise, distortion, and loud overlapping background chatter."
        ]
        
        noise_inputs = processor(text=noise_prompts, return_tensors="pt", padding=True)
        noise_outputs = model.get_text_features(**noise_inputs)
        
        if isinstance(noise_outputs, torch.Tensor):
            noise_embeds = noise_outputs
        elif hasattr(noise_outputs, "text_embeds") and noise_outputs.text_embeds is not None:
            noise_embeds = noise_outputs.text_embeds
        elif hasattr(noise_outputs, "pooler_output"):
            noise_embeds = noise_outputs.pooler_output
        else:
            noise_embeds = noise_outputs[0]
            
        if noise_embeds.shape[-1] != model.config.projection_dim:
            noise_embeds = model.text_projection(noise_embeds)
            
        mean_noise_embed = torch.mean(noise_embeds, dim=0, keepdim=True)
        
        # 3. Vector Math: Subtract 25% of the noise signature from the audio
        purified_audio_embed = mean_audio_embed - (0.25 * mean_noise_embed)
        
        # Re-normalize for cosine similarity
        audio_embed = F.normalize(purified_audio_embed, p=2, dim=-1)
    
    best_match = None
    highest_sim = -1.0
    
    for language, prompts in EKSTEP_PROMPTS.items():
        text_embed = get_averaged_text_embeddings(model, processor, prompts)
        similarity = torch.mm(audio_embed, text_embed.T).item()
        
        if similarity > highest_sim:
            highest_sim = similarity
            best_match = language
            
    return best_match, highest_sim