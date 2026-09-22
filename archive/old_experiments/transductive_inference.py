import os
import argparse
import numpy as np
import torch
import torch.nn.functional as F
from msclap import CLAP
from prompts import EKSTEP_PROMPTS
from sklearn.metrics import accuracy_score

def get_text_embeddings(clap_model, language_names):
    """
    Get text embeddings for languages using prompt ensembling.
    Uses prompts from EKSTEP_PROMPTS and adds a generic prompt.
    """
    print(f"Extracting text embeddings for {len(language_names)} languages...")
    text_embeddings = []
    
    for lang in language_names:
        # Get specific prompts if they exist, otherwise use an empty list
        prompts = EKSTEP_PROMPTS.get(lang, []).copy()
        # Add a simple generic prompt
        prompts.append(f"A person speaking {lang}.")
        
        # Extract embeddings for all prompts for this language
        with torch.no_grad():
            prompt_embeddings = clap_model.get_text_embeddings(prompts)
            # Average the embeddings for prompt ensembling
            avg_embedding = prompt_embeddings.mean(dim=0)
            text_embeddings.append(avg_embedding)
            
    return torch.stack(text_embeddings)

def transductive_zero_shot(X_test, text_embeddings, n_iterations=5, alpha=0.5, temperature=1.0):
    """
    TransCLIP-style transductive inference.
    
    Args:
        X_test: (N, D) tensor of audio embeddings
        text_embeddings: (C, D) tensor of text prompt embeddings (C = num classes)
        n_iterations: number of refinement iterations
        alpha: weight for combining text prototypes with estimated centroids (0=text only, 1=centroids only)
        temperature: softmax temperature for soft assignments
    
    Returns:
        predictions: (N,) tensor of predicted class indices
        probabilities: (N, C) tensor of prediction probabilities
    """
    # Normalize embeddings
    X_test_norm = F.normalize(X_test, p=2, dim=-1)
    prototypes = F.normalize(text_embeddings, p=2, dim=-1)
    
    for iteration in range(n_iterations):
        # Step 1: Compute similarities
        similarities = X_test_norm @ prototypes.T  # (N, C)
        
        # Step 2: Soft assignments with temperature scaling
        soft_assignments = F.softmax(similarities / temperature, dim=1)  # (N, C)
        
        # Step 3: Estimate class centroids from test data
        # Weighted average of test features, weighted by soft assignments
        centroids = soft_assignments.T @ X_test_norm  # (C, D)
        centroids = F.normalize(centroids, p=2, dim=-1)
        
        # Step 4: Update prototypes as weighted combination of text and centroids
        prototypes = (1 - alpha) * F.normalize(text_embeddings, p=2, dim=-1) + alpha * centroids
        prototypes = F.normalize(prototypes, p=2, dim=-1)
    
    # Final predictions
    final_similarities = X_test_norm @ prototypes.T
    predictions = torch.argmax(final_similarities, dim=1)
    probabilities = F.softmax(final_similarities / temperature, dim=1)
    
    return predictions, probabilities

def run_experiment(features_file, split_mode, run_grid_search):
    # Ensure reproducible splits
    np.random.seed(42)
    
    # Load features
    print(f"Loading features from {features_file}...")
    data = np.load(features_file)
    X = data['X']
    y = data['y']
    
    # Convert to PyTorch tensors
    X_tensor = torch.tensor(X, dtype=torch.float32)
    
    # Unique languages
    all_languages = sorted(list(set(y)))
    print(f"Found {len(all_languages)} languages in the dataset.")
    
    # Initialize CLAP
    clap_model = CLAP(version='2023', use_cuda=torch.cuda.is_available())
    
    # Split handling
    if split_mode == 'language_split':
        print("\n--- Language Split Evaluation (Seen/Unseen) ---")
        # Split languages randomly using seed 42
        shuffled_langs = all_languages.copy()
        np.random.shuffle(shuffled_langs)
        split_idx = len(shuffled_langs) // 2
        seen_langs = set(shuffled_langs[:split_idx])
        unseen_langs = set(shuffled_langs[split_idx:])
        
        print(f"Seen languages: {seen_langs}")
        print(f"Unseen languages: {unseen_langs}")
        
        # Filter test data to only include unseen languages
        unseen_mask = np.isin(y, list(unseen_langs))
        X_test = X_tensor[unseen_mask]
        y_test = y[unseen_mask]
        
        # Test vocabulary is only the unseen languages
        test_langs = sorted(list(unseen_langs))
        test_text_embeddings = get_text_embeddings(clap_model, test_langs)
    else:
        print("\n--- Full Dataset Evaluation ---")
        X_test = X_tensor
        y_test = y
        test_langs = all_languages
        test_text_embeddings = get_text_embeddings(clap_model, test_langs)
        
    print(f"Test set size: {len(X_test)} samples over {len(test_langs)} languages.")
    
    # Convert true labels to indices based on test_langs
    lang_to_idx = {lang: i for i, lang in enumerate(test_langs)}
    y_test_idx = np.array([lang_to_idx[l] for l in y_test])
    
    # Baseline Inductive Zero-Shot
    print("\nRunning Inductive (Standard) Zero-Shot Baseline...")
    X_test_norm = F.normalize(X_test, p=2, dim=-1)
    proto_norm = F.normalize(test_text_embeddings, p=2, dim=-1)
    sims = X_test_norm @ proto_norm.T
    preds = torch.argmax(sims, dim=1).numpy()
    
    baseline_acc = accuracy_score(y_test_idx, preds)
    print(f"Baseline Inductive Zero-Shot Accuracy: {baseline_acc * 100:.2f}%")
    
    results = [
        f"Evaluation Mode: {split_mode}",
        f"Baseline Inductive Zero-Shot Accuracy: {baseline_acc * 100:.2f}%",
        "--- Transductive Results ---"
    ]
    
    best_acc = -1
    best_config = None
    
    # Grid search or default
    if run_grid_search:
        print("\nRunning Transductive Grid Search...")
        iterations_list = [1, 3, 5, 10]
        alpha_list = [0.1, 0.3, 0.5, 0.7]
        temp_list = [0.01, 0.05, 0.1, 0.5, 1.0]
    else:
        print("\nRunning Transductive Inference with default parameters...")
        iterations_list = [5]
        alpha_list = [0.5]
        temp_list = [1.0]
        
    for iters in iterations_list:
        for alpha in alpha_list:
            for temp in temp_list:
                pred_t, _ = transductive_zero_shot(
                    X_test, test_text_embeddings, 
                    n_iterations=iters, alpha=alpha, temperature=temp
                )
                pred_t = pred_t.numpy()
                acc = accuracy_score(y_test_idx, pred_t)
                
                config_str = f"iters={iters}, alpha={alpha}, temp={temp}"
                result_str = f"Config: [{config_str}] -> Accuracy: {acc * 100:.2f}%"
                print(result_str)
                results.append(result_str)
                
                if acc > best_acc:
                    best_acc = acc
                    best_config = config_str
                    
    summary = f"\nBest Transductive Config: [{best_config}] -> Accuracy: {best_acc * 100:.2f}%"
    print(summary)
    results.append(summary)
    
    with open("transductive_results.txt", "w") as f:
        f.write("\n".join(results) + "\n")
    print("Results saved to transductive_results.txt")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="TransCLIP-style transductive inference for CLAP.")
    parser.add_argument("--features-file", type=str, default="clap_features.npz", help="Path to features file")
    parser.add_argument("--split", type=str, choices=['full', 'language_split'], default='language_split', help="Evaluation split mode")
    parser.add_argument("--grid-search", action="store_true", help="Run full hyperparameter grid search")
    
    args = parser.parse_args()
    
    if not os.path.exists(args.features_file):
        print(f"Error: Features file {args.features_file} not found.")
        print("Please extract features first or provide the correct path.")
        exit(1)
        
    run_experiment(args.features_file, args.split, args.grid_search)
