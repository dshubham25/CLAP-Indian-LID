import torch
from msclap import CLAP
import numpy as np
from tqdm import tqdm
from preprocess import get_dataloaders
from coop_clap import CoOpCLAP
from cocoop_clap import CoCoOpCLAP
from palm_clap import PALM_CLAP
import argparse

SEEN_LANGS = ['Marathi', 'Kannada', 'English', 'Odia', 'Punjabi', 'Malayalam', 'Telugu', 'Gujarati']
UNSEEN_LANGS = ['Hindi', 'Assamese', 'Bengali', 'Tamil']

def evaluate_model(model_name, device):
    print(f"Evaluating {model_name} on UNSEEN zero-shot languages: {UNSEEN_LANGS}")
    
    if model_name == "coop":
        model = CoOpCLAP(n_ctx=4, classnames=UNSEEN_LANGS, device=device).to(device)
        model.ctx.data = torch.load("coop_learned_prompts.pt").to(device)
    elif model_name == "cocoop":
        model = CoCoOpCLAP(n_ctx=4, classnames=UNSEEN_LANGS, device=device).to(device)
        state = torch.load("cocoop_learned_prompts.pt")
        model.ctx.data = state['ctx'].to(device)
        model.meta_net.load_state_dict(state['meta_net'])
    elif model_name == "palm":
        model = PALM_CLAP(classnames=UNSEEN_LANGS, device=device).to(device)
        model.prompt_adapter.load_state_dict(torch.load("palm_adapter.pt"))
    else:
        raise ValueError("Invalid model name")
        
    model.eval()
    
    # Load UNSEEN dataset
    _, _, unseen_loader = get_dataloaders(batch_size=32)
    
    correct = 0
    total = 0
    
    with torch.no_grad():
        for batch in tqdm(unseen_loader):
            audio_files = batch['audio_file']
            labels = batch['label'].to(device)
            
            # Map original UNSEEN labels (8, 9, 10, 11) to (0, 1, 2, 3) for this batch
            labels = labels - 8
            
            logits = model(audio_files)
            preds = logits.argmax(dim=1)
            
            correct += (preds == labels).sum().item()
            total += labels.size(0)
            
    acc = correct / total
    print(f"\nZero-Shot Accuracy on 4 UNSEEN Languages ({model_name.upper()}): {acc*100:.2f}%")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, choices=["coop", "cocoop", "palm"], required=True)
    args = parser.parse_args()
    
    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    evaluate_model(args.model, device)
