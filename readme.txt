# Handwritten Digit Recognition (CNN + ViT)

**Student:** Usman Rasheed  
**Roll No:** TRN 2026 000313  
**Module:** AI Engineering Bootcamp — Module 04 Sunday Project

## Overview

A complete handwritten digit recognition system that trains **two models side by side** — a Convolutional Neural Network (CNN) and a Vision Transformer (ViT) — on the `sklearn.datasets.load_digits()` 8×8 dataset, then deploys both in a **Streamlit web app** that accepts uploaded photos and shows predictions from each model.

## Features

- **CNN:** 2 conv layers + 2 fully-connected layers → 10-class logits
- **ViT:** 16 patches of 2×2 → patch embedding (32-dim) + positional encoding + TransformerEncoder (2 layers, 4 heads) → 10-class head
- **Fair comparison:** identical stratified train/valid/test split (60/20/20) for both models
- **Upload pipeline:** Pillow → grayscale → invert bright background → crop → center → resize 8×8 → ÷255 → predict
- **Error handling:** rejects missing, blank, and unreadable files without crashing
- **Fresh-launch ready:** loads saved checkpoints; no retraining inside the app

## Results (test set)

| Model | Test Accuracy | Macro F1 |
|---|---|---|
| Majority baseline | 0.10 | — |
| CNN | 0.9444 | 0.9445 |
| ViT | 0.9278 | 0.9279 |

## Project Structure
