import os
import glob
import numpy as np
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F

# ======== Load MFCC Features and Labels ========
features = []
labels = []

# Wakeword = 1
for file_path in glob.glob("data/features_wakeword/*.npy"):
    mfcc = np.load(file_path)
    mfcc = (mfcc - np.mean(mfcc)) / np.std(mfcc)  # Normalize
    features.append(mfcc)
    labels.append(1)

# Non-wakeword = 0
for file_path in glob.glob("data/features_nonwakewords/*.npy"):
    mfcc = np.load(file_path)
    mfcc = (mfcc - np.mean(mfcc)) / np.std(mfcc)  # Normalize
    features.append(mfcc)
    labels.append(0)

labels = np.array(labels)

# ======== Train-Test Split ========
features_train, features_test, labels_train, labels_test = train_test_split(
    features, labels, test_size=0.33, random_state=42
)

# ======== Dataset Class ========
class WakeWordDataset(Dataset):
    def __init__(self, features, labels):
        self.features = features
        self.labels = labels

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        feature = torch.tensor(self.features[idx], dtype=torch.float32)
        label = torch.tensor(self.labels[idx], dtype=torch.long)
        return feature, label

# ======== CNN Model Definition with Dynamic FC Layer ========
class WakeWordModel(nn.Module):
    def __init__(self, input_shape):
        super(WakeWordModel, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)

        # Dynamically compute the flattened size
        with torch.no_grad():
            dummy_input = torch.randn(1, *input_shape).unsqueeze(1)  # (1, 1, time, n_mfcc)
            x = self.pool(F.relu(self.conv1(dummy_input)))
            x = self.pool(F.relu(self.conv2(x)))
            flattened_size = x.view(1, -1).shape[1]

        self.fc1 = nn.Linear(flattened_size, 64)
        self.fc2 = nn.Linear(64, 2)  # Binary classification

    def forward(self, x):
        x = x.unsqueeze(1)  # Add channel dimension
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        return self.fc2(x)

# ======== Prepare DataLoaders ========
train_dataset = WakeWordDataset(features_train, labels_train)
test_dataset = WakeWordDataset(features_test, labels_test)

train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=4)

# ======== Initialize Model, Loss, Optimizer ========
input_shape = features_train[0].shape  # e.g., (63, 13)
model = WakeWordModel(input_shape=input_shape)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# ======== Training Loop ========
epochs = 10
for epoch in range(epochs):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for inputs, labels in train_loader:
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, preds = torch.max(outputs, 1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    train_acc = 100 * correct / total
    print(f"Epoch {epoch+1}, Loss: {running_loss / len(train_loader):.4f}, Train Accuracy: {train_acc:.2f}%")

# ======== Evaluation ========
model.eval()
correct = 0
total = 0
with torch.no_grad():
    for inputs, labels in test_loader:
        outputs = model(inputs)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print(f"Test Accuracy: {100 * correct / total:.2f}%")

torch.save(model.state_dict(), 'wakeword_model.pth')
print("Model saved to wakeword_model.pth")