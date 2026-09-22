import numpy as np
import random
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

def run_knn_language_split(features_file="clap_features.npz"):
    print(f"Loading features from {features_file}...")
    data = np.load(features_file)
    X = data['X']
    y = data['y']
    
    # Identify unique languages
    unique_langs = list(set(y))
    unique_langs.sort()
    
    # 70% Seen / 30% Unseen Language partition
    random.seed(42) # Ensure reproducible splits
    random.shuffle(unique_langs)
    split_idx = int(0.7 * len(unique_langs))
    
    seen_langs = set(unique_langs[:split_idx])
    unseen_langs = set(unique_langs[split_idx:])
    
    print(f"Languages assigned to KNN Training (Seen): {sorted(list(seen_langs))}")
    print(f"Languages hidden from KNN Training (Unseen): {sorted(list(unseen_langs))}\n")
    
    # Filter the dataset into Seen vs Unseen
    seen_mask = np.array([label in seen_langs for label in y])
    unseen_mask = np.array([label in unseen_langs for label in y])
    
    X_seen, y_seen = X[seen_mask], y[seen_mask]
    X_unseen, y_unseen = X[unseen_mask], y[unseen_mask]
    
    # We still need to test the model on the languages it DID see to ensure it learned them
    # Split the "Seen" languages 70/30 into train and test
    X_train, X_test_seen, y_train, y_test_seen = train_test_split(
        X_seen, y_seen, test_size=0.3, random_state=42, stratify=y_seen
    )
    
    # Train the KNN classifier ONLY on Seen languages
    print("Training KNN on Seen language data...")
    knn = KNeighborsClassifier(n_neighbors=5, metric='cosine')
    knn.fit(X_train, y_train)
    
    # 1. Evaluate performance on Seen languages
    preds_seen = knn.predict(X_test_seen)
    acc_seen = accuracy_score(y_test_seen, preds_seen)
    print(f"\nKNN Accuracy on Seen Languages Test Split: {acc_seen * 100:.2f}%")
    
    # 2. Evaluate performance on Unseen languages
    if len(X_unseen) > 0:
        preds_unseen = knn.predict(X_unseen)
        acc_unseen = accuracy_score(y_unseen, preds_unseen)
        print(f"KNN Accuracy on Unseen Languages Test Split: {acc_unseen * 100:.2f}%")
        print("\nNote: Supervised classifiers mathematically achieve 0% accuracy on unseen labels because they cannot output a class (like 'Tamil') if it was never in their 'y_train' labels.")

if __name__ == "__main__":
    run_knn_language_split()