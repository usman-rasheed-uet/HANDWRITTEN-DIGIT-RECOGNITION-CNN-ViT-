# Handwritten Digit Recognition (CNN + ViT)

**Student:** Usman Rasheed  
**Roll No:** TRN 2026 000313  
**Module:** AI Engineering Bootcamp — Module 04 Sunday Project

## 📌 Overview

A complete handwritten digit recognition system that trains **two distinct models side by side** — a Convolutional Neural Network (CNN) and a Vision Transformer (ViT) — on the `sklearn.datasets.load_digits()` 8×8 dataset. 

The project includes a **Streamlit web app** that accepts uploaded photos of handwritten digits, processes them using the exact same pipeline as training, and displays predictions from both models.

## ✨ Features

- **CNN:** 2 convolutional layers + 2 fully-connected layers → 10-class logits.
- **ViT:** 16 patches of 2×2 → patch embedding (32-dim) + positional encoding + TransformerEncoder (2 layers, 4 heads) → 10-class head.
- **Fair Comparison:** Identical stratified train/valid/test split (60/20/20) used for both models without leakage.
- **Robust Upload Pipeline:** Pillow → grayscale → invert bright background → crop to strokes → center → resize 8×8 → divide by 255 → predict.
- **Error Handling:** Gracefully rejects missing, blank, and unreadable files without crashing.
- **Fresh-Launch Ready:** Loads saved checkpoints; performs zero training inside the app.

## 📊 Results (Test Set)

| Model | Test Accuracy | Macro F1 |
|---|---|---|
| Majority Baseline | 0.10 | — |
| **CNN** | **0.9444** | **0.9445** |
| **ViT** | **0.9278** | **0.9279** |

## 📂 Project Structure

```text
├── app.py                      # Streamlit application
├── cnn_digit.pt                # Trained CNN weights (checkpoint)
├── vit_digit.pt                # Trained ViT weights (checkpoint)
├── preprocess_meta.json        # Preprocessing settings (pixel divisors, class map)
├── Digit_Project.ipynb         # Training and evaluation notebook
├── evaluation_report.md        # Test accuracy, F1, confusion matrices
├── README.md                   # This file
├── make_test_images_real.py    # Script that generated the 200 test images
├── sample_for_app.png          # Guaranteed demo image
└── test_images/                # 200 real handwritten digits + blank/broken negatives
