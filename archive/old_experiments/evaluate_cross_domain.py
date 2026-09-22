import torch
from msclap import CLAP
import numpy as np
from tqdm import tqdm
from coop_clap import CoOpCLAP, AudioDataset
from cocoop_clap import CoCoOpCLAP
from palm_clap import PALM_CLAP
import argparse
from torch.utils.data import DataLoader

SEEN_LANGS = ['Marathi', 'Kannada', 'English', 'Odia', 'Punjabi', 'Malayalam', 'Telugu', 'Gujarati']

def evaluate_model(model_name, dataset_path, device):
    print(f"Evaluating {model_name.upper()} on CROSS-DOMAIN Dataset: {dataset_path}")
    print(f"Target Languages: {SEEN_LANGS}")
    
    if model_name == "coop":
        model = CoOpCLAP(n_ctx=4, classnames=SEEN_LANGS, device=device).to(device)
        model.ctx.data = torch.load("coop_learned_prompts.pt").to(device)
    elif model_name == "cocoop":
        model = CoCoOpCLAP(n_ctx=4, classnames=SEEN_LANGS, device=device).to(device)
        state = torch.load("cocoop_learned_prompts.pt")
        model.ctx.data = state['ctx'].to(device)
        model.meta_net.load_state_dict(state['meta_net'])
    elif model_name == "palm":
        model = PALM_CLAP(classnames=SEEN_LANGS, device=device).to(device)
        model.prompt_adapter.load_state_dict(torch.load("palm_adapter.pt"))
    else:
        raise ValueError("Invalid model name")
        
    model.eval()
    
    # Load Cross-Domain Dataset (IITMandi_YouTube)
    # Reusing AudioDataset which automatically filters to SEEN_LANGS
    test_dataset = AudioDataset(dataset_path, SEEN_LANGS)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    
    correct = 0
    total = 0
    
    with torch.no_grad():
        for audio_files, labels in tqdm(test_loader):
            labels = labels.to(device)
            
            logits = model(audio_files)
            preds = logits.argmax(dim=1)
            
            correct += (preds == labels).sum().item()
            total += labels.size(0)
            
    acc = correct / total
    print(f"\nCross-Domain Accuracy on {dataset_path} ({model_name.upper()}): {acc*100:.2f}%")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, choices=["coop", "cocoop", "palm"], required=True)
    parser.add_argument("--dataset", type=str, default="IITMandi_YouTube")
    args = parser.parse_args()
    
    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    evaluate_model(args.model, args.dataset, device)
