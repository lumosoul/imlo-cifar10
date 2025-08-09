import os
from tqdm import tqdm
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import random_split

"""
CIFAR-10 Evaluation/Training Script (IMLO Coursework)
Purpose:
    Train and evaluate a custom CNN on the official CIFAR-10 splits and report
    validation/test performance. CPU-only execution. No external data or pretrained
    weights.

References (IEEE):
[1] A. Krizhevsky, “Learning Multiple Layers of Features from Tiny Images,” 2009.
[2] A. Paszke et al., “PyTorch: An Imperative Style, High-Performance Deep Learning Library,”
    NeurIPS, 2019.
[3] Torchvision CIFAR-10 documentation:
    https://pytorch.org/vision/main/generated/torchvision.datasets.CIFAR10.html
"""

# ----------------------------------------------------------------------
# Configuration
# ----------------------------------------------------------------------
BASE_DIR = os.path.abspath(os.path.dirname(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "models")
os.makedirs(MODEL_DIR, exist_ok=True)

MODEL_NAME = "cnn_1"
MODEL_PATH = os.path.join(MODEL_DIR, f"{MODEL_NAME}.pth")

# Coursework rule: maximum 4 hours training on CPU; GPU not allowed.
# Uncomment below to enable GPU if permitted:
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device = "cpu"
print(f"Device: {device}")

# ----------------------------------------------------------------------
# Data
# ----------------------------------------------------------------------
"""
Data augmentation is permitted provided that images are derived only from the
CIFAR-10 training set, with no external images or prior knowledge.
"""

train_transform = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomCrop(32, padding=4),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

valid_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

"""
Dataset:
    CIFAR-10 is loaded using torchvision's dataset utilities to ensure official
    splits and correct labels. No external datasets are used.
"""

train_dataset = torchvision.datasets.CIFAR10(
    root="./data", train=True, download=True, transform=train_transform
)

# Split training set into train/validation (80/20)
train_size = int(0.8 * len(train_dataset))
val_size = len(train_dataset) - train_size
train_subset, val_subset = random_split(train_dataset, [train_size, val_size])

train_loader = torch.utils.data.DataLoader(train_subset, batch_size=100, shuffle=True)
valid_loader = torch.utils.data.DataLoader(val_subset, batch_size=100, shuffle=True)

test_dataset = torchvision.datasets.CIFAR10(
    root="./data", train=False, download=True, transform=valid_transform
)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=100, shuffle=False)

print(train_dataset.classes)
# ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

# ----------------------------------------------------------------------
# Model
# ----------------------------------------------------------------------
"""
Neural Network Architecture:
    Two convolutional layers, followed by three fully connected layers.
    Dropout applied for regularisation.
"""

class ConvNet(nn.Module):
    def __init__(self):
        super(ConvNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.bn1 = nn.BatchNorm2d(6)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.bn2 = nn.BatchNorm2d(16)
        self.dropout = nn.Dropout(0.5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

model = ConvNet().to(device)

# ----------------------------------------------------------------------
# Optimisation
# ----------------------------------------------------------------------
epochs = 30
min_valid_loss = np.inf
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
n_total_steps = len(train_loader)

# ----------------------------------------------------------------------
# Training & Validation
# ----------------------------------------------------------------------
for epoch in range(epochs):
    train_acc = 0.0
    train_loss = 0.0

    for images, labels in tqdm(train_loader):
        images = images.to(device)
        labels = labels.to(device)

        outputs = model(images)
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        acc = (outputs.argmax(dim=1) == labels).float().mean()
        train_acc += acc

    train_acc = train_acc / len(train_loader) * 100
    train_loss = train_loss / len(train_loader)

    # Validation
    model.eval()
    valid_acc = 0.0
    valid_loss = 0.0
    for images, labels in tqdm(valid_loader):
        images = images.to(device)
        labels = labels.to(device)
        outputs = model(images)
        loss = criterion(outputs, labels)
        valid_loss += loss.item()
        acc = (outputs.argmax(dim=1) == labels).float().mean()
        valid_acc += acc

    valid_acc = valid_acc / len(valid_loader) * 100
    prev_valid_loss = valid_loss
    valid_loss = valid_loss / len(valid_loader)

    print(
        f"Epoch {epoch+1} | "
        f"Train Acc: {train_acc:.2f}% | Train Loss: {train_loss:.6f} | "
        f"Valid Acc: {valid_acc:.2f}% | Valid Loss: {valid_loss:.6f}"
    )

    if min_valid_loss > valid_loss:
        print(
            f"Validation loss improved: {min_valid_loss:.6f} → {valid_loss:.6f}. Saving checkpoint."
        )
        min_valid_loss = valid_loss
        torch.save(model.state_dict(), MODEL_PATH)
    print()

print("Training complete.")

# ----------------------------------------------------------------------
# Test Evaluation
# ----------------------------------------------------------------------
with torch.no_grad():
    n_correct = 0
    n_samples = 0
    n_class_correct = [0 for _ in range(10)]
    n_class_samples = [0 for _ in range(10)]

    for images, labels in test_loader:
        images = images.to(device)
        labels = labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)

        n_samples += labels.size(0)
        n_correct += (predicted == labels).sum().item()

        for i in range(labels.size(0)):
            label = labels[i]
            pred = predicted[i]
            if label == pred:
                n_class_correct[label] += 1
            n_class_samples[label] += 1

    total_acc = 100.0 * n_correct / n_samples
    print(f"Total accuracy: {total_acc:.2f}%")

    for i in range(10):
        class_acc = 100.0 * n_class_correct[i] / n_class_samples[i]
        print(f"Accuracy of class {train_dataset.classes[i]}: {class_acc:.2f}%")

"""
Example results:
Total accuracy: 65.65%
Accuracy of class airplane: 66.80%
Accuracy of class automobile: 81.50%
Accuracy of class bird: 53.00%
Accuracy of class cat: 36.20%
Accuracy of class deer: 62.70%
Accuracy of class dog: 60.20%
Accuracy of class frog: 72.70%
Accuracy of class horse: 71.00%
Accuracy of class ship: 74.30%
Accuracy of class truck: 78.10%
"""
