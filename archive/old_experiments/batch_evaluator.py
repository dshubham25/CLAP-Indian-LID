import os
import glob
from evaluator import load_clap_model, predict_language

FOLDER_TO_LANG = {
    "hin": "Hindi", "ben": "Bengali", "mar": "Marathi", "tel": "Telugu",
    "tam": "Tamil", "guj": "Gujarati", "urd": "Urdu", "kan": "Kannada",
    "ori": "Odia", "odi": "Odia", "mal": "Malayalam", "pun": "Punjabi",
    "asm": "Assamese", "mai": "Maithili", "sat": "Santali", "kas": "Kashmiri",
    "nep": "Nepali", "snd": "Sindhi", "kok": "Konkani", "doi": "Dogri",
    "mni": "Manipuri", "brx": "Bodo", "san": "Sanskrit", "eng": "English",
}

def get_processed_files(results_file):
    processed = set()
    if os.path.exists(results_file):
        with open(results_file, "r") as f:
            for line in f:
                if "|" in line:
                    file_path = line.split("|")[0].strip()
                    processed.add(file_path)
    return processed

def run_batch_evaluation(dataset_root, results_file):
    model, processor = load_clap_model()
    wav_files = glob.glob(os.path.join(dataset_root, "*", "*.wav"))
    processed_files = get_processed_files(results_file)
    total = 0
    correct = 0

    # Count previous correct/total if resuming
    if os.path.exists(results_file):
        with open(results_file, "r") as f:
            for line in f:
                # Use split to check the exact final word
                if "|" in line:
                    total += 1
                    status = line.split("|")[-1].strip()
                    if status == "MATCH":
                        correct += 1

    # Filter out files we've already done
    files_to_process = [f for f in wav_files if f not in processed_files]
    total_files_to_process = len(files_to_process)
    
    print(f"\nFound {len(wav_files)} total files.")
    print(f"{len(processed_files)} already processed. {total_files_to_process} remaining to process.\n")

    for i, file_path in enumerate(files_to_process, 1):
        folder = os.path.basename(os.path.dirname(file_path))
        true_lang = FOLDER_TO_LANG.get(folder)
        
        if true_lang is None:
            print(f"Skipping {file_path}: Unknown language code '{folder}'")
            continue

        print(f"[{i}/{total_files_to_process}] Analyzing: {os.path.basename(file_path)} ...", end=" ", flush=True)

        pred_lang, sim_score = predict_language(file_path, model, processor)
        match = (pred_lang == true_lang)
        status = "MATCH" if match else "MISMATCH"
        
        result_str = f"{file_path} | True: {true_lang} | Pred: {pred_lang} | Similarity: {sim_score:.4f} | {status}"
        print(f"-> {status} ({pred_lang})")

        total += 1
        if match:
            correct += 1

        # 4. Append IMMEDIATELY to the text file. This saves progress file-by-file.
        with open(results_file, "a") as f:
            f.write(result_str + "\n")

    # Print summary when loop completes
    if total == 0:
        print("No .wav files found for evaluation.")
    else:
        accuracy = 100.0 * correct / total
        print(f"Total files: {total}")
        print(f"Correct predictions: {correct}")
        print(f"Baseline Accuracy: {accuracy:.2f}%")


        
        with open(results_file, "a") as f:
            f.write(f"\nTotal files: {total}\n")
            f.write(f"Correct predictions: {correct}\n")
            f.write(f"Baseline Accuracy: {accuracy:.2f}%\n")

if __name__ == "__main__":
    DATASET_ROOTS = [
        ("IITMandi_YouTube", "batch_results_ensemble.txt")
    ]
    for dataset_root, results_file in DATASET_ROOTS:
        print(f"\nEvaluating {dataset_root} ...")
        run_batch_evaluation(dataset_root, results_file)