# AD_inital_RestNET — Alzheimer's Disease Detection Using ResNet50

A deep learning pipeline for detecting and classifying Alzheimer's Disease severity from brain MRI scans using transfer learning with ResNet50.

## Overview

This project implements a **4-class classification** system that analyzes brain MRI images to identify:

| Class | Description |
|-------|-------------|
| **NonDemented** | Normal / healthy |
| **VeryMildDemented** | Early-stage Alzheimer's |
| **MildDemented** | Intermediate-stage Alzheimer's |
| **ModerateDemented** | Advanced-stage Alzheimer's |

The pipeline includes data cleaning with brain contour extraction, offline data augmentation for class balancing, a two-phase training strategy, comprehensive evaluation metrics, Grad-CAM interpretability visualizations, and an interactive Gradio web interface for inference.

## Model Architecture

**Base model:** ResNet50 (pre-trained on ImageNet)

```
Input (224×224×3 RGB MRI)
    ↓
Data Augmentation (RandomFlip, RandomRotation, RandomZoom)
    ↓
ResNet50 Preprocessing
    ↓
ResNet50 Base
    ↓
GlobalAveragePooling2D
    ↓
Dropout (0.3)
    ↓
Dense (4 units, softmax) → class probabilities
```

### Two-Phase Training Strategy

1. **Phase 1 — Transfer Learning:** The ResNet50 base is frozen; only the custom classification head is trained with a learning rate of `0.001`.
2. **Phase 2 — Fine-Tuning:** Layers from index 143 onward are unfrozen and the entire model is trained with a lower learning rate of `1e-5`, using early stopping (patience 5) and model checkpointing.

## Project Structure

```
AD_inital_RestNET/
├── README.md               # Project documentation
├── AdvMJP.ipynb            # Main notebook — complete ML pipeline (12 blocks)
└── Project Synopsis.pdf    # Project proposal and objectives
```

### Notebook Pipeline (AdvMJP.ipynb)

| Block | Purpose |
|-------|---------|
| 1 | Imports, Google Drive mount, seed setup (42) |
| 2 | Data cleaning — brain contour extraction and 224×224 resizing via OpenCV |
| 3 | Offline data augmentation for class balancing (rotation, zoom, flip) |
| 4 | Dataset loading with `image_dataset_from_directory` (batch size 32) |
| 5 | Phase 1 — transfer learning with frozen ResNet50 |
| 6 | Phase 2 — fine-tuning deep layers with callbacks |
| 7 | Evaluation — confusion matrix, classification report, Cohen's Kappa, MCC |
| 8 | Grad-CAM — gradient-weighted activation maps for interpretability |
| 9 | Visualization — ROC curves, learning curves, performance plots |
| 10 | Statistical analysis — confidence intervals, per-class metrics |
| 11 | Dependency installation (Gradio) |
| 12 | Interactive Gradio web interface for predictions |

## Requirements

### Dependencies

- Python 3.x
- TensorFlow / Keras
- OpenCV (`opencv-python`)
- NumPy
- Pandas
- scikit-learn
- Matplotlib
- Seaborn
- Gradio
- Google Colab (for Drive integration and GPU access)

### Hardware

A GPU is recommended for training. The notebook is designed to run on **Google Colab** with a **T4 GPU**.

### Dataset

The pipeline expects an Alzheimer's MRI dataset organized by class on Google Drive:

```
Alzheimer_dataset/
├── NonDemented/
├── VeryMildDemented/
├── MildDemented/
└── ModerateDemented/
```

## Usage

1. Upload the MRI dataset to your Google Drive under `Alzheimer_dataset/`.
2. Open `AdvMJP.ipynb` in [Google Colab](https://colab.research.google.com/).
3. Select a GPU runtime (**Runtime → Change runtime type → T4 GPU**).
4. Run the cells sequentially:
   - Blocks 1–3 prepare and clean the data.
   - Blocks 4–6 train the model in two phases.
   - Blocks 7–10 evaluate performance and generate visualizations.
   - Blocks 11–12 launch the Gradio web interface for interactive predictions.

## Key Features

- **Brain contour extraction** — OpenCV-based preprocessing isolates brain tissue from MRI scans.
- **Data augmentation** — Offline augmentation addresses class imbalance.
- **Transfer learning** — Leverages ImageNet-trained ResNet50 features.
- **Grad-CAM** — Visual explanations highlighting regions influencing predictions.
- **Comprehensive metrics** — Accuracy, precision, recall, F1-score, Cohen's Kappa, MCC, ROC curves.
- **Interactive UI** — Gradio web interface for uploading MRI images and viewing predictions.
- **Reproducibility** — Fixed random seed (42) across all libraries.

## Configuration

| Parameter | Value |
|-----------|-------|
| Image size | 224 × 224 |
| Batch size | 32 |
| Random seed | 42 |
| Phase 1 learning rate | 0.001 |
| Phase 2 learning rate | 1e-5 |
| Fine-tune from layer | 143 |
| Early stopping patience | 5 |

## Disclaimer

This project is intended for **educational and research purposes only**. It is not validated for clinical use and should not be used for medical diagnosis.