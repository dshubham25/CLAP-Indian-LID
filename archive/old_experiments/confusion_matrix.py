import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, accuracy_score

def generate_confusion_matrix(features_file="clap_features.npz"):
    print(f"Loading features from {features_file}...")
    try:
        data = np.load(features_file)
        X = data['X']
        y = data['y']
    except FileNotFoundError:
        print(f"Error: {features_file} not found. Please run the feature extraction script first.")
        return

    # 70/30 Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
    
    print("Training KNN Classifier...")
    knn = KNeighborsClassifier(n_neighbors=5, metric='cosine')
    knn.fit(X_train, y_train)
    
    print("Predicting test set...")
    y_pred = knn.predict(X_test)
    
    print(f"Accuracy: {accuracy_score(y_test, y_pred) * 100:.2f}%")
    
    # Generate Confusion Matrix
    print("Generating Confusion Matrix Plot...")
    labels = sorted(list(set(y))) # Alphabetical order for consistency
    cm = confusion_matrix(y_test, y_pred, labels=labels)
    
    # Plotting
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels)
    plt.title('KNN Confusion Matrix (Stratified 70/30 Split)', fontsize=16)
    plt.ylabel('True Language', fontsize=12)
    plt.xlabel('Predicted Language', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    
    # Save the plot
    plot_filename = "confusion_matrix.png"
    plt.savefig(plot_filename, dpi=300)
    print(f"Confusion Matrix successfully saved as '{plot_filename}' in your current directory!")

if __name__ == "__main__":
    generate_confusion_matrix()