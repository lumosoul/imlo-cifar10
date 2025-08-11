# Convolutional Neural Networks for CIFAR-10 Classification

This repository contains multiple implementations of Convolutional Neural Networks (CNNs) for image classification on the [CIFAR-10 dataset](https://www.cs.toronto.edu/~kriz/cifar.html).  
Each model is trained and evaluated using PyTorch, with progressive improvements across different versions.

---

## Installation

### 1. Create and activate a virtual environment
python -m venv myenv  
myenv\Scripts\activate.bat   # On Windows  
# source myenv/bin/activate  # On Linux/MacOS

### 2. Install dependencies
pip install -r requirements.txt

---

## Project Structure

/tests                # Directory with experimental CNN variants  
/tests/models         # Trained model weights for each variant
/models               # Trained model weights for each variant
train.py              # Training script (no arguments required)  
test.py               # Evaluation script (no arguments required)  

- train.py — trains the selected CNN on the official CIFAR-10 training set.  
- test.py — evaluates the trained model on the CIFAR-10 test set.  
  Note: Run test.py only after full training, or place cnn_4.pth inside models/. Otherwise, model loading will fail.

---

## Model Variants and Results

### CNN #1 — Baseline Architecture
File: cnn_1.py  
- Two convolutional layers  
- Three fully connected layers  
- Parameters inspired by standard practice exercises  

Results:  
Overall accuracy: 65.65%  
airplane: 66.80%  
automobile: 81.50%  
bird: 53.00%  
cat: 36.20%  
deer: 62.70%  
dog: 60.20%  
frog: 72.70%  
horse: 71.00%  
ship: 74.30%  
truck: 78.10%  

---

### CNN #2 — MaxPooling Enhancement
File: cnn_2.py  
- Added an additional pooling layer (nn.MaxPool2d(2, 2)) after the second convolution.

Results:  
Overall accuracy: 66.23%  
airplane: 59.20%  
automobile: 80.70%  
bird: 62.60%  
cat: 50.00%  
deer: 56.40%  
dog: 45.70%  
frog: 76.20%  
horse: 72.40%  
ship: 88.90%  
truck: 70.20%  

Observation: Slight improvement in overall accuracy.

---

### CNN #3 — Augmentation and Architecture Refinement
File: cnn_3.py  
- Removed transforms.RandomCrop(32, padding=4) to preserve image details.  
- Added RandomRotation(15), ColorJitter(), and RandomGrayscale() for data augmentation.  
- Added a third convolutional layer, reduced convolution kernel size from 5×5 to smaller filters.  
- Reduced fully connected layers for efficiency.  

Results:  
Overall accuracy: 66.86%  
airplane: 65.40%  
automobile: 84.70%  
bird: 50.10%  
cat: 48.20%  
deer: 66.00%  
dog: 55.70%  
frog: 77.00%  
horse: 70.30%  
ship: 80.10%  
truck: 71.10%  

Observation: Overall accuracy increased slightly. Shows promising direction for dataset-specific tuning.

---

### CNN #4 — Final Optimized Model
File: cnn_4.py  
- Validation set split from training data (80/20).  
- Optimized architecture with batch normalization and dropout.  
- Advanced data augmentation pipeline.  
- Learning rate scheduling applied.  

Results:  
Overall accuracy: 81.16%

Per-class performance (with common misclassifications):  

airplane — 85.70% (ship: 3.60, automobile: 1.70, bird: 2.70)  
automobile — 91.00% (truck: 5.80, ship: 0.90, airplane: 0.70)  
bird — 73.30% (deer: 7.10, airplane: 4.10, frog: 4.10)  
cat — 63.20% (dog: 12.30, bird: 6.10, deer: 5.70)  
deer — 81.40% (horse: 4.70, bird: 6.00, cat: 3.00)  
dog — 69.50% (cat: 15.20, horse: 4.50, bird: 3.60)  
frog — 84.10% (cat: 4.90, bird: 3.90, deer: 2.90)  
horse — 88.80% (deer: 3.20, dog: 2.20, cat: 2.20)  
ship — 86.10% (airplane: 5.50, automobile: 2.90)  
truck — 88.50% (automobile: 4.90, airplane: 1.80)  

---

## How to Run

### Train
python train.py

### Test
python test.py  
Note: Ensure cnn_4.pth is present in the models/ directory before running test.py.

---

## Summary
The iterative development from CNN #1 to CNN #4 improved classification accuracy from 65.65% to 81.16%.  
Key factors included better data augmentation, optimized convolutional kernels, additional layers, and learning rate scheduling.
