# Vision Transformer for Skin Lesion Classification

**Transfer Learning | Medical Image Classification **

## 1. Description

- Developed a multiclass skin lesion classification pipeline using a pretrained **Vision Transformer (ViT-Base/32)** and transfer learning.
- Applied a two-stage fine-tuning strategy to adapt ImageNet-21k pretrained representations to dermatoscopic images.
- Investigated class imbalance, overfitting, and minority-class misclassification using validation metrics, confusion matrices, and attention maps.
- Explored transformer-based feature learning for medical image classification.

## 2. Dataset

- **Dataset:** [HAM10000 — Human Against Machine with 10,000 Training Images](https://doi.org/10.1038/sdata.2018.161)
- **Task:** Seven-class skin lesion classification.
- **Labels:** Melanocytic nevi (`nv`), melanoma (`mel`), benign keratosis (`bkl`), basal cell carcinoma (`bcc`), actinic keratoses (`akiec`), vascular lesions (`vasc`), and dermatofibroma (`df`).
- **Data characteristics:** Approximately 10,000 dermatoscopic images with substantial class imbalance; melanocytic nevi form the majority class.
- **Data split:** 80% training, 10% validation, and 10% testing, as documented in the coursework.

## 3. Preprocessing

- Resized input images to **224 × 224 pixels** to match the pretrained model's input configuration.
- Normalized image pixel values using the mean and standard deviation specified by the ViT image processor.
- Applied training-time data augmentation using horizontal flips, small rotations, and colour jitter.
- Prepared the classification head for seven output classes.
- Used macro F1-score alongside accuracy to evaluate performance under class imbalance.

## 4. Model Architecture

- **Backbone:** ImageNet-21k pretrained ViT-Base/32 from Hugging Face.
- **Patch embedding:** Divided each input image into non-overlapping 32 × 32 patches and projected them into token embeddings.
- **Transformer encoder:** Used 12 encoder layers with multi-head self-attention, feed-forward networks, residual connections, and layer normalization.
- **Classification:** Extracted the `[CLS]` token representation and passed it through a task-specific classification head for seven-class prediction.
- **Transfer-learning strategy:**
  - **Stage 1 — Feature adaptation:** Froze the pretrained transformer backbone and trained the new classification head.
  - **Stage 2 — Fine-tuning:** Unfroze the final transformer blocks and fine-tuned them using a smaller learning rate.

## 5. Training and Validation

- **Framework:** Hugging Face Transformers Trainer API.
- **Optimizer:** AdamW.
- **Loss function:** Cross-entropy loss.
- **Evaluation metrics:** Accuracy and macro F1-score.
- **Stage 1:** Trained the classification head for 15 epochs with the backbone frozen.
- **Stage 2:** Configured fine-tuning for up to 20 epochs; early stopping occurred at epoch 10 to limit overfitting.
- **Validation monitoring:** Compared training and validation loss and evaluated changes in accuracy and macro F1-score across both stages.

## 6. Results

| Experiment | Accuracy | Macro F1-score |
|---|---:|---:|
| Stage 1 — Frozen backbone | 0.73 | 0.35 |
| Stage 2 — Fine-tuned backbone | 0.83 | 0.62 |
| **Improvement** | **+0.10** | **+0.27** |

### Key Findings

- Fine-tuning improved accuracy by **10 percentage points** and macro F1-score by **0.27**, indicating better overall classification performance.
- The gap between accuracy and macro F1-score highlights the difficulty of classifying underrepresented lesion categories.
- Confusion matrix analysis revealed misclassification between minority classes and the dominant `nv` class, including confusion between melanoma and melanocytic nevi.
- Training–validation loss divergence indicated overfitting, motivating early stopping.
- Visualised attention maps using final-layer `[CLS]`-to-patch attention, averaged across attention heads, to inspect the model's spatial attention patterns.

### Limitations and Future Work

- Investigate class-weighted loss and weighted random sampling to improve minority-class recognition.
- Evaluate per-class precision, recall, F1-score, and test-set performance to obtain a more complete assessment.
- Explore calibration and additional interpretability methods to understand prediction reliability.
- Treat the model as an experimental research implementation, **not a clinically validated diagnostic system**.

## 7. References

1. Dosovitskiy, A. et al. (2021). *An Image is Worth 16×16 Words: Transformers for Image Recognition at Scale.* International Conference on Learning Representations (ICLR). [Paper](https://arxiv.org/abs/2010.11929)
2. Tschandl, P., Rosendahl, C., and Kittler, H. (2018). *The HAM10000 Dataset, a Large Collection of Multi-Source Dermatoscopic Images of Common Pigmented Skin Lesions.* Scientific Data, 5, 180161. [Paper](https://doi.org/10.1038/sdata.2018.161)
3. Hugging Face. *Transformers Documentation.* [Documentation](https://huggingface.co/docs/transformers)
4. **Implementation:** [Machine-Learning-Tutorial — GitHub Repository](https://github.com/sri-lekha-19/Machine-Learning-Tutorial)
5. **Experiment notebook:** [Google Colab — Model Implementation](https://colab.research.google.com/drive/1KL5EdLBBwsPIy7aZMjit-qDqybWtfpAw?usp=sharing)
