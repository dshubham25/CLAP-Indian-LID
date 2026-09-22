import os
import glob
import types
import torch
import soundfile as sf
import torchaudio.transforms as T
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, accuracy_score
from msclap import CLAP
from collections import defaultdict

FOLDER_TO_LANG = {
    "hin": "Hindi", "ben": "Bengali", "mar": "Marathi", "tel": "Telugu",
    "tam": "Tamil", "guj": "Gujarati", "urd": "Urdu", "kan": "Kannada",
    "ori": "Odia", "odi": "Odia", "mal": "Malayalam", "pun": "Punjabi",
    "eng": "English"
}

LANG_FAMILY = {
    "Hindi": "Indo-Aryan", "Bengali": "Indo-Aryan", "Marathi": "Indo-Aryan",
    "Gujarati": "Indo-Aryan", "Urdu": "Indo-Aryan", "Punjabi": "Indo-Aryan",
    "Odia": "Indo-Aryan",
    "Telugu": "Dravidian", "Tamil": "Dravidian", "Kannada": "Dravidian",
    "Malayalam": "Dravidian",
    "English": "Germanic"
}

# Order languages by family for grouped confusion matrix
FAMILY_ORDER = [
    "Hindi", "Marathi", "Gujarati", "Punjabi", "Odia",  # Indo-Aryan
    "Kannada", "Telugu", "Malayalam",                       # Dravidian
    "English"                                               # Germanic
]

SEEN_LANGS = ['Marathi', 'Kannada', 'English', 'Odia', 'Punjabi', 'Malayalam', 'Telugu', 'Gujarati']
# Reorder SEEN_LANGS by family
SEEN_LANGS_ORDERED = [l for l in FAMILY_ORDER if l in SEEN_LANGS]

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

def evaluate_dataset(clap_model, text_embeddings, dataset_path, dataset_name):
    files = glob.glob(os.path.join(dataset_path, "**", "*.wav"), recursive=True)
    
    # Filter to only seen languages
    valid_files = []
    for f in files:
        folder = os.path.basename(os.path.dirname(f))
        if folder in FOLDER_TO_LANG and FOLDER_TO_LANG[folder] in SEEN_LANGS:
            valid_files.append(f)
    
    print(f"  Found {len(valid_files)} valid files in {dataset_name}")
    
    true_labels = []
    pred_labels = []
    per_lang_correct = defaultdict(int)
    per_lang_total = defaultdict(int)
    
    for i, f in enumerate(valid_files):
        true_lang = FOLDER_TO_LANG[os.path.basename(os.path.dirname(f))]
        try:
            emb = clap_model.get_audio_embeddings([f], resample=True).detach()
            sims = clap_model.compute_similarity(emb, text_embeddings).squeeze(0).detach().cpu().numpy()
            pred_idx = np.argmax(sims)
            pred_lang = SEEN_LANGS_ORDERED[pred_idx]
            
            true_labels.append(true_lang)
            pred_labels.append(pred_lang)
            per_lang_total[true_lang] += 1
            if pred_lang == true_lang:
                per_lang_correct[true_lang] += 1
        except Exception as e:
            pass
        
        if (i + 1) % 100 == 0:
            print(f"    Processed {i+1}/{len(valid_files)}...")
    
    return true_labels, pred_labels, per_lang_correct, per_lang_total

def run_phase1():
    print("=" * 60)
    print("PHASE 1: Per-Language Accuracy & Family Confusion Matrix")
    print("=" * 60)
    
    print("\nInitializing MS-CLAP...")
    clap_model = CLAP(version='2023', use_cuda=False)
    clap_model.read_audio = types.MethodType(custom_read_audio, clap_model)
    
    prompts = [f"A person speaking in {lang}" for lang in SEEN_LANGS_ORDERED]
    print("Extracting Text Prompt Embeddings...")
    text_embeddings = clap_model.get_text_embeddings(prompts).detach()
    
    # --- Evaluate In-Domain ---
    print("\n--- Evaluating In-Domain (Ekstep) ---")
    id_true, id_pred, id_correct, id_total = evaluate_dataset(
        clap_model, text_embeddings, "ekstep_seen", "Ekstep (In-Domain)")
    
    # --- Evaluate Cross-Domain ---
    print("\n--- Evaluating Cross-Domain (IITMandi YouTube) ---")
    cd_true, cd_pred, cd_correct, cd_total = evaluate_dataset(
        clap_model, text_embeddings, "IITMandi_YouTube", "IITMandi (Cross-Domain)")
    
    # =====================================================
    # STEP 1.1: Per-Language Accuracy Table
    # =====================================================
    print("\n" + "=" * 80)
    print("STEP 1.1: Per-Language Accuracy Breakdown")
    print("=" * 80)
    print(f"{'Language':<15} {'Family':<15} {'In-Domain %':<15} {'Cross-Domain %':<15} {'Drop %':<10}")
    print("-" * 70)
    
    table_data = []
    for lang in SEEN_LANGS_ORDERED:
        family = LANG_FAMILY[lang]
        id_acc = (id_correct[lang] / id_total[lang] * 100) if id_total[lang] > 0 else 0
        cd_acc = (cd_correct[lang] / cd_total[lang] * 100) if cd_total[lang] > 0 else 0
        drop = id_acc - cd_acc
        print(f"{lang:<15} {family:<15} {id_acc:<15.1f} {cd_acc:<15.1f} {drop:<10.1f}")
        table_data.append((lang, family, id_acc, cd_acc, drop))
    
    overall_id = accuracy_score(id_true, id_pred) * 100
    overall_cd = accuracy_score(cd_true, cd_pred) * 100
    print("-" * 70)
    print(f"{'OVERALL':<15} {'':<15} {overall_id:<15.1f} {overall_cd:<15.1f} {overall_id - overall_cd:<10.1f}")
    
    # Compute family-level averages
    print("\n" + "=" * 80)
    print("Family-Level Average Accuracy Drop")
    print("=" * 80)
    family_drops = defaultdict(list)
    for lang, family, id_acc, cd_acc, drop in table_data:
        family_drops[family].append(drop)
    
    for family in ["Indo-Aryan", "Dravidian", "Germanic"]:
        if family in family_drops:
            avg_drop = np.mean(family_drops[family])
            print(f"  {family:<15} Avg Drop: {avg_drop:.1f}%")
    
    # =====================================================
    # STEP 1.2: Language-Family Grouped Confusion Matrices
    # =====================================================
    print("\nGenerating Confusion Matrices...")
    
    fig, axes = plt.subplots(1, 2, figsize=(22, 9))
    
    # In-Domain Confusion Matrix
    cm_id = confusion_matrix(id_true, id_pred, labels=SEEN_LANGS_ORDERED)
    cm_id_pct = cm_id.astype('float') / cm_id.sum(axis=1)[:, np.newaxis] * 100
    
    # Add family separators
    family_labels = [f"{l}\n({LANG_FAMILY[l][:5]})" for l in SEEN_LANGS_ORDERED]
    
    sns.heatmap(cm_id_pct, annot=True, fmt='.0f', cmap='Blues',
                xticklabels=family_labels, yticklabels=family_labels, ax=axes[0],
                vmin=0, vmax=100, linewidths=0.5)
    axes[0].set_title("In-Domain (Ekstep) - % Confusion", fontsize=14, fontweight='bold')
    axes[0].set_xlabel("Predicted Language")
    axes[0].set_ylabel("True Language")
    
    # Cross-Domain Confusion Matrix
    cm_cd = confusion_matrix(cd_true, cd_pred, labels=SEEN_LANGS_ORDERED)
    cm_cd_pct = cm_cd.astype('float') / cm_cd.sum(axis=1)[:, np.newaxis] * 100
    
    sns.heatmap(cm_cd_pct, annot=True, fmt='.0f', cmap='Reds',
                xticklabels=family_labels, yticklabels=family_labels, ax=axes[1],
                vmin=0, vmax=100, linewidths=0.5)
    axes[1].set_title("Cross-Domain (YouTube) - % Confusion", fontsize=14, fontweight='bold')
    axes[1].set_xlabel("Predicted Language")
    axes[1].set_ylabel("True Language")
    
    # Draw family group boundaries
    indo_aryan_count = len([l for l in SEEN_LANGS_ORDERED if LANG_FAMILY[l] == "Indo-Aryan"])
    dravidian_start = indo_aryan_count
    dravidian_count = len([l for l in SEEN_LANGS_ORDERED if LANG_FAMILY[l] == "Dravidian"])
    
    for ax in axes:
        ax.axhline(y=indo_aryan_count, color='black', linewidth=3)
        ax.axvline(x=indo_aryan_count, color='black', linewidth=3)
        ax.axhline(y=indo_aryan_count + dravidian_count, color='black', linewidth=3)
        ax.axvline(x=indo_aryan_count + dravidian_count, color='black', linewidth=3)
    
    plt.tight_layout()
    plt.savefig("phase1_family_confusion.png", dpi=150, bbox_inches='tight')
    plt.close()
    
    # =====================================================
    # Bar chart: Per-language drop
    # =====================================================
    fig, ax = plt.subplots(figsize=(12, 6))
    langs = [t[0] for t in table_data]
    id_accs = [t[2] for t in table_data]
    cd_accs = [t[3] for t in table_data]
    
    x = np.arange(len(langs))
    width = 0.35
    
    colors_id = ['#1565C0' if LANG_FAMILY[l] == 'Indo-Aryan' else '#2E7D32' if LANG_FAMILY[l] == 'Dravidian' else '#F57F17' for l in langs]
    colors_cd = ['#64B5F6' if LANG_FAMILY[l] == 'Indo-Aryan' else '#81C784' if LANG_FAMILY[l] == 'Dravidian' else '#FFF176' for l in langs]
    
    bars1 = ax.bar(x - width/2, id_accs, width, color=colors_id, edgecolor='black', linewidth=0.5)
    bars2 = ax.bar(x + width/2, cd_accs, width, color=colors_cd, edgecolor='black', linewidth=0.5)
    
    ax.set_xlabel('Language', fontsize=13)
    ax.set_ylabel('Accuracy (%)', fontsize=13)
    ax.set_title('Per-Language Accuracy: In-Domain vs Cross-Domain', fontsize=15, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels([f"{l}\n({LANG_FAMILY[l][:5]})" for l in langs], fontsize=10)
    ax.set_ylim(0, 105)
    
    # Add value labels on bars
    for bar in bars1:
        height = bar.get_height()
        ax.annotate(f'{height:.0f}', xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=8)
    for bar in bars2:
        height = bar.get_height()
        ax.annotate(f'{height:.0f}', xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=8)
    
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='#1565C0', label='Indo-Aryan (In-Domain)'),
        Patch(facecolor='#64B5F6', label='Indo-Aryan (Cross-Domain)'),
        Patch(facecolor='#2E7D32', label='Dravidian (In-Domain)'),
        Patch(facecolor='#81C784', label='Dravidian (Cross-Domain)'),
    ]
    ax.legend(handles=legend_elements, loc='upper right', fontsize=9)
    
    plt.tight_layout()
    plt.savefig("phase1_per_language_accuracy.png", dpi=150, bbox_inches='tight')
    plt.close()
    
    print("\nDone! Saved:")
    print("  - phase1_family_confusion.png")
    print("  - phase1_per_language_accuracy.png")

if __name__ == "__main__":
    run_phase1()
