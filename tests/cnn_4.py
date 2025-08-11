"""
CNN Model 4 for CIFAR-10 Classification
=======================================

This script defines, trains, validates, and evaluates a Convolutional Neural Network (CNN)
on the CIFAR-10 dataset using PyTorch. It includes learning rate scheduling and advanced
data augmentation. The best model is saved when validation loss improves.

"""

import os
from tqdm import tqdm
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import random_split
import torch.optim.lr_scheduler as lr_scheduler

# ----------------------------------------------------------------------
# Paths and device configuration
# ----------------------------------------------------------------------
BASE_DIR = os.path.abspath(os.path.dirname(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "models")
os.makedirs(MODEL_DIR, exist_ok=True)
MODEL_NAME = "cnn_4"
MODEL_PATH = os.path.join(MODEL_DIR, f"{MODEL_NAME}.pth")

epochs = 30
total_iters = 100
device = 'cpu'
print(f"Device: {device}")

# ----------------------------------------------------------------------
# Data preprocessing and loading
# ----------------------------------------------------------------------
train_transform = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ColorJitter(),
    transforms.RandomGrayscale(),
    transforms.ToTensor(),
    transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5))
])

valid_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5))
])

train_dataset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=train_transform)
train_size = int(0.8 * len(train_dataset))
val_size = len(train_dataset) - train_size
train_subset, val_subset = random_split(train_dataset, [train_size, val_size])

train_loader = torch.utils.data.DataLoader(train_subset, batch_size=100, shuffle=True)
valid_loader = torch.utils.data.DataLoader(val_subset, batch_size=100, shuffle=True)

test_dataset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=valid_transform)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=100, shuffle=False)

print(train_dataset.classes)

# ----------------------------------------------------------------------
# Model definition
# ----------------------------------------------------------------------
class ConvNet(nn.Module):
    def __init__(self):
        super(ConvNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.bn2 = nn.BatchNorm2d(64)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.bn3 = nn.BatchNorm2d(128)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(8192, 256)
        self.bn4 = nn.BatchNorm1d(256)
        self.fc2 = nn.Linear(256, 128)
        self.bn5 = nn.BatchNorm1d(128)
        self.dropout = nn.Dropout(0.3)
        self.fc3 = nn.Linear(128, 10)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.pool2(F.relu(self.bn2(self.conv2(x))))
        x = self.pool3(F.relu(self.bn3(self.conv3(x))))
        x = self.flatten(x)
        x = F.relu(self.bn4(self.fc1(x)))
        x = F.relu(self.bn5(self.fc2(x)))
        x = self.dropout(x)
        x = self.fc3(x)
        return x

# ----------------------------------------------------------------------
# Training configuration
# ----------------------------------------------------------------------
model = ConvNet().to(device)
min_valid_loss = np.inf
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
scheduler = lr_scheduler.LinearLR(optimizer, start_factor=1.0, end_factor=0.1, total_iters=total_iters)
n_total_steps = len(train_loader)

# ----------------------------------------------------------------------
# Training and validation loop
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
        acc = ((outputs.argmax(dim=1) == labels).float().mean())
        train_acc += acc
    train_acc = train_acc / len(train_loader) * 100
    prev_train_loss = train_loss
    train_loss = train_loss / len(train_loader)

    valid_acc = 0.0
    valid_loss = 0.0
    model.eval()
    for images, labels in tqdm(valid_loader):
        images = images.to(device)
        labels = labels.to(device)
        target = model(images)
        loss = criterion(target, labels)
        valid_loss += loss.item()
        acc = ((target.argmax(dim=1) == labels).float().mean())
        valid_acc += acc
    valid_acc = valid_acc / len(test_loader) * 100
    prev_valid_loss = valid_loss
    valid_loss = valid_loss / len(test_loader)

    before_lr = optimizer.param_groups[0]["lr"]
    scheduler.step()
    after_lr = optimizer.param_groups[0]["lr"]

    print(f'Epoch {epoch+1} | Train Acc: {train_acc:.2f}% | Train Loss: {train_loss:.6f} | '
          f'Valid Acc: {valid_acc:.2f}% | Valid Loss: {valid_loss:.6f}')
    print(f"Learning rate: {before_lr:.5f} -> {after_lr:.5f}")

    if min_valid_loss > valid_loss:
        print(f'Validation loss decreased: ({min_valid_loss:.6f}--->{valid_loss:.6f}) \t Saving model')
        min_valid_loss = valid_loss
        torch.save(model.state_dict(), MODEL_PATH)
    print()

print('Training completed!')

# ----------------------------------------------------------------------
# Test set evaluation
# ----------------------------------------------------------------------
model.load_state_dict(torch.load(MODEL_PATH))
model.eval()
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
    acc = 100.0 * n_correct / n_samples
    print(f'Overall accuracy: {acc:.2f}%')
    for i in range(10):
        acc = 100.0 * n_class_correct[i] / n_class_samples[i]
        print(f'Accuracy for class {train_dataset.classes[i]}: {acc:.2f}%')
