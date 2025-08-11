"""
CNN Model 4 Test Script for CIFAR-10 Classification
===================================================

This script loads a trained CNN model for CIFAR-10 and evaluates it on the test dataset.
It reports overall accuracy, per-class accuracy, and misclassification statistics.

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

# ----------------------------------------------------------------------
# Paths and device configuration
# ----------------------------------------------------------------------
BASE_DIR = os.path.abspath(os.path.dirname(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "models")
os.makedirs(MODEL_DIR, exist_ok=True)
MODEL_NAME = "cnn_4"
MODEL_PATH = os.path.join(MODEL_DIR, f"{MODEL_NAME}.pth")

device = 'cpu'
print(f"Device: {device}")

# ----------------------------------------------------------------------
# Data preprocessing and loading
# ----------------------------------------------------------------------
valid_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5))
])

test_dataset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=valid_transform)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=100, shuffle=False)

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
# Model loading
# ----------------------------------------------------------------------
model = ConvNet().to(device)
other_classes = {i: {} for i in range(10)}
model.load_state_dict(torch.load(MODEL_PATH))
model.eval()

# ----------------------------------------------------------------------
# Evaluation on test set
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
            else:
                other_classes_stat = other_classes[int(label)]
                try:
                    other_classes_stat[int(pred)] += 1
                except KeyError:
                    other_classes_stat[int(pred)] = 1
                other_classes[int(label)] = other_classes_stat
            n_class_samples[label] += 1

    acc = 100.0 * n_correct / n_samples
    print(f'Overall accuracy: {acc:.2f}%')

    for i in range(10):
        acc = 100.0 * n_class_correct[i] / n_class_samples[i]
        print(f'Accuracy for class {test_dataset.classes[i]}: {acc:.2f}%')

        other_classes_stat = other_classes[i]
        misclassification_rate = 0
        for j in range(10):
            try:
                class_acc = 100.0 * other_classes_stat[j] / n_class_samples[i]
                misclassification_rate += class_acc
                print(f'- Misclassified as {test_dataset.classes[j]}: {class_acc:.2f}%')
            except KeyError:
                pass
        print(f"Total misclassified: {misclassification_rate:.2f}%\n")
