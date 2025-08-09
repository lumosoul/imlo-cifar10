# CIFAR-10 Convolutional Neural Network Classifiers

## Abstract
This repository contains two custom convolutional neural network (CNN) architectures developed for the **Intelligent Systems: Machine Learning and Optimisation (IMLO)** coursework.  
Both models are trained **from scratch** on the official CIFAR-10 dataset using CPU-only execution and evaluated on the corresponding test split.  
The project fully complies with coursework requirements: no external datasets, pretrained weights, or GPU acceleration were used.

---

## 1. Environment Setup

It is recommended to create a Python virtual environment before installing dependencies.

    python -m venv myenv
    # Windows
    myenv\Scripts\activate.bat
    # macOS / Linux
    source myenv/bin/activate

    pip install -r requirements.txt

---

## 2. Project Structure

    /src             # Source code: CNN architectures and training/evaluation scripts
        cnn_1.py     # CNN architecture #1
        cnn_2.py     # CNN architecture #2
    /models          # Trained model weight files (.pth)
    requirements.txt # Python dependencies
    README.md        # Project documentation

---

## 3. Model Descriptions and Results

### 3.1 CNN #1
**File:** `src/cnn_1.py`  
**Architecture:** Two convolutional layers with batch normalization and max pooling, followed by three fully connected layers with dropout regularisation.

**Test Accuracy:**
Airplane     — 66.80%  
Automobile   — 81.50%  
Bird         — 53.00%  
Cat          — 36.20%  
Deer         — 62.70%  
Dog          — 60.20%  
Frog         — 72.70%  
Horse        — 71.00%  
Ship         — 74.30%  
Truck        — 78.10%  
**Total**    — 65.65%

---

### 3.2 CNN #2
**File:** `src/cnn_2.py`  
**Modification:** Added a second pooling layer after the second convolutional block.

**Test Accuracy:**
Airplane     — 59.20%  
Automobile   — 80.70%  
Bird         — 62.60%  
Cat          — 50.00%  
Deer         — 56.40%  
Dog          — 45.70%  
Frog         — 76.20%  
Horse        — 72.40%  
Ship         — 88.90%  
Truck        — 70.20%  
**Total**    — 66.23%

---

## 4. Comparative Analysis

| Class       | CNN #1 (%) | CNN #2 (%) | Δ Accuracy |
|-------------|------------|------------|------------|
| Airplane    | 66.80      | 59.20      | -7.60      |
| Automobile  | 81.50      | 80.70      | -0.80      |
| Bird        | 53.00      | 62.60      | +9.60      |
| Cat         | 36.20      | 50.00      | +13.80     |
| Deer        | 62.70      | 56.40      | -6.30      |
| Dog         | 60.20      | 45.70      | -14.50     |
| Frog        | 72.70      | 76.20      | +3.50      |
| Horse       | 71.00      | 72.40      | +1.40      |
| Ship        | 74.30      | 88.90      | +14.60     |
| Truck       | 78.10      | 70.20      | -7.90      |
| **Total**   | **65.65**  | **66.23**  | **+0.58**  |

**Summary:**  
The second pooling layer in CNN #2 produced a marginal overall improvement (+0.58%), with notable gains for the *cat*, *bird*, and *ship* classes. However, some classes experienced reduced accuracy, indicating class-dependent trade-offs.

---

## 5. Academic Integrity
All code in this repository was written by the author in compliance with the IMLO coursework rules:
- No external datasets or pretrained weights were used.
- All experiments adhered to the official CIFAR-10 dataset split.
- Training was conducted exclusively on CPU within the permitted time constraints.

---

## 6. References
[1] A. Krizhevsky, “Learning Multiple Layers of Features from Tiny Images,” 2009.  
[2] A. Paszke et al., “PyTorch: An Imperative Style, High-Performance Deep Learning Library,” *Advances in Neural Information Processing Systems*, 2019.  
[3] Torchvision CIFAR-10 documentation: https://pytorch.org/vision/main/generated/torchvision.datasets.CIFAR10.html