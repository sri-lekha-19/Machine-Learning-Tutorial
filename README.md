# Machine-Learning-Tutorial
A short demonstration of Vision Transformers on Skin Lesion Classification using Transfer Learning technique

Full Implementation
Code File: https://colab.research.google.com/drive/1KL5EdLBBwsPIy7aZMjit-qDqybWtfpAw?usp=sharing

# Vision Transformer for Skin Lesion Classification

## Overview

This project applies a **Vision Transformer (ViT)** model to classify skin lesions using the **HAM10000 dataset**. Transfer learning is used to handle the limited dataset size, and attention maps are generated to interpret model predictions.

---

## Dataset

* HAM10000 skin lesion dataset
* 7 classes: nv, mel, bkl, bcc, akiec, vasc, df
* Imbalanced dataset (nv is dominant)

---

## Method

* Pretrained ViT (ViT-Base 32) from Hugging Face Open Source Models
* Two-stage Transfer Learning:

  * Stage 1: Train classification head
  * Stage 2: Fine-tune transformer layers
* Image size: 224×224
* Data augmentation: flip, rotation, color jitter

---

## Results

* Accuracy: ~0.86
* Macro F1-score: ~0.69

Model performs well on majority classes but struggles with minority classes due to imbalance.

---

## Interpretability

Attention maps show that the model focuses on lesion regions, indicating meaningful feature learning.


## Summary

This project demonstrates the effectiveness of Vision Transformers for medical image classification using transfer learning, along with interpretability through attention visualization.

