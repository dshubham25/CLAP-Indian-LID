# linear_probe.py
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, random_split
from sklearn.metrics import classification_report

LANG_TO_FOLDER = {
    0: "Hindi", 1: "Bengali", 2: "Marathi", 3: "Telugu", 4: "Tamil", 
    5: "Gujarati", 6: "Urdu", 7: "Kannada", 8: "Odia", 9: "Malayalam", 
    10: "Punjabi", 11: "Assamese", 12: "Maithili", 13: "Santali", 
    14: "Kashmiri", 15: "Nepali", 16: "Sindhi", 17: "Konkani", 
    18: "Dogri", 19: "Manipuri", 20: "Bodo", 21: "Sanskrit", 22: "English"
}

class LinearProbe(nn.Module):
    def __init__(self, input_dim=512, num_classes=23):
        super(LinearProbe, self).__init__()
        self.classifier = nn.Linear(input_dim, num_classes)

    def forward(self, x):
        return self.classifier(x)

def train_and_evaluate():
    print("Loading IITMandi_YouTube embeddings for In-Domain Evaluation...")
    # Load ONLY the noisy YouTube dataset
    data = torch.load("test_features.pt")
    X, y = data['embeddings'], data['labels']
    
    total_files = len(X)
    
    # 80% Train, 20% Test Split
    train_size = int(0.8 * total_files)
    test_size = total_files - train_size
    
    full_dataset = TensorDataset(X, y)
    train_dataset, test_dataset = random_split(full_dataset, [train_size, test_size])
    
    print(f"Total Files: {total_files}")
    print(f"Training on {train_size} files, Testing on {test_size} files.\n")

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    model = LinearProbe(input_dim=512, num_classes=23)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    epochs = 300
    print(f"Training Linear Head for {epochs} Epochs...")
    
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for inputs, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            
        if (epoch + 1) % 10 == 0:
            print(f"Epoch [{epoch+1}/{epochs}] | Loss: {running_loss/len(train_loader):.4f}")

    print("\nEvaluating on 20% Unseen IITMandi_YouTube Test Set...")
    model.eval()
    
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for inputs, labels in test_loader:
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            
            all_preds.extend(predicted.numpy())
            all_labels.extend(labels.numpy())

    correct = sum(p == l for p, l in zip(all_preds, all_labels))
    accuracy = 100.0 * correct / test_size
    
    print(f"\n=========================================")
    print(f"IN-DOMAIN LINEAR PROBING ACCURACY: {accuracy:.2f}%")
    print(f"=========================================\n")
    
    # Only show names for classes that actually exist in the test split
    present_classes = sorted(list(set(all_labels)))
    target_names = [LANG_TO_FOLDER[i] for i in present_classes]
    
    print(classification_report(all_labels, all_preds, target_names=target_names, zero_division=0))

if __name__ == "__main__":
    train_and_evaluate()