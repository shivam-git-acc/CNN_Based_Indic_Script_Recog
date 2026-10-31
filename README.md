# ✍️ Multi-Script Indic Handwritten Character Recognition

**SMAI Assignment 3 — IIIT Hyderabad**

A deep-learning pipeline that identifies the script of a handwritten character image (Devanagari, Tamil, Bengali, or Kannada) and then predicts the exact character — all displayed as the real Unicode glyph.

Live demo → [Hugging Face Space](https://huggingface.co/spaces/KGan31/SMAI-A3-Demo)

---

## Table of Contents

- [Overview](#overview)
- [Pipeline](#pipeline)
- [Model Architectures](#model-architectures)
- [Datasets](#datasets)
- [Project Structure](#project-structure)
- [Setup & Installation](#setup--installation)
- [Running the Streamlit App](#running-the-streamlit-app)
- [End-to-End Accuracy Testing](#end-to-end-accuracy-testing)
- [Results](#results)
- [Deployment](#deployment)

---

## Overview

The system performs **two-stage classification**:

1. **Script Routing** — A Router CNN classifies the input image into one of four scripts.
2. **Character Classification** — The script-specific CNN predicts the exact character class, which is then mapped to its Unicode representation.

Supported scripts:

| Script      | Language   | Classes |
|-------------|------------|---------|
| Devanagari  | Hindi      | 46      |
| Tamil       | Tamil      | 156     |
| Bengali     | Bengali    | 84      |
| Kannada     | Kannada    | 10      |

---

## Pipeline

```
Input Image
     │
     ▼
┌─────────────┐
│  Router CNN │  ── predicts script (Devanagari / Tamil / Bengali / Kannada)
└─────────────┘
     │
     ▼
┌──────────────────────┐
│  Script-Specific CNN │  ── predicts character class index
└──────────────────────┘
     │
     ▼
Class index → Unicode label map → Display character (e.g. "क", "அ", "ক", "೧")
```

Preprocessing (identical for router and character models):
- Convert to grayscale
- Resize to target size (32×32 for all scripts, 28×28 for Kannada)
- Normalise to `[-1, 1]` via `(pixel / 255 - 0.5) / 0.5`

---

## Model Architectures

### Router CNN
- 3 convolutional blocks (32 → 64 → 128 channels)
- BatchNorm + ReLU + MaxPool in each block
- Spatial Dropout (0.1 / 0.2)
- Fully-connected head: 128×4×4 → 256 → 4
- Input: 32×32 grayscale

### Devanagari CNN (`ModelA`)
- 3 conv blocks: 64 → 128 → 256 channels (5×5, 3×3, 3×3 kernels)
- BatchNorm + ReLU + MaxPool per block
- FC: 256×4×4 → 512 → 46
- Input: 32×32 grayscale

### Bengali CNN (`BengaliCNN`)
- 4 conv blocks: 32 → 32 → 64 → 64 → 128 → 128 channels
- Spatial Dropout (0.1 / 0.15)
- **AdaptiveAvgPool2d(1)** → removes spatial dimensions
- FC: 128 → 256 → 84
- Input: 32×32 grayscale

### Tamil CNN (`TamilCNN`)
- 4 conv blocks: 32 → 64 → 128 → 256 channels
- Spatial Dropout (0.1 / 0.15 / 0.2 / 0.25)
- **AdaptiveAvgPool2d(1)**
- FC: 256 → 512 → 156
- Input: 32×32 grayscale

### Kannada CNN (`KannadaCNN`)
- 3 conv blocks: 32 → 64 → 128 channels (5×5, 3×3, 3×3 kernels)
- BatchNorm (momentum=0.15) + Spatial Dropout (0.25) per block
- FC: 128×7×7 → 256 → 10
- Input: **28×28** grayscale

---

## Datasets

| Script     | Dataset                         | Train / Test Split |
|------------|---------------------------------|--------------------|
| Devanagari | Devanagari Character Dataset    | From notebook      |
| Tamil      | uTHCD (uTHCD_a 70-30 split)     | 70 % / 30 %        |
| Bengali    | Bengali Handwritten Dataset     | From notebook      |
| Kannada    | Kannada-MNIST                   | train.csv / test.csv |

**Router dataset**: [Router Dataset](https://www.kaggle.com/datasets/kgan31/router-dataset)
- 20 000 images total (5 000 per script)
- Pre-organized into `{script}/{class}/` folders
- Metadata in `manifest.json`

---

## Project Structure

```
A3/
├── dataset/
│   ├── Router_Dataset/
│   │   └── router_dataset/         # Router test images
│   │       ├── devanagari/         # 46 class folders
│   │       ├── tamil/              # 156 class folders
│   │       ├── bengali/            # 84 class folders
│   │       ├── kannada/            # 10 class folders
│   │       └── manifest.json       # Dataset metadata
│   ├── Tamil/
│   │   └── uTHCD_a(70-30-split)/   # Tamil train/test images + CSVs
│   └── Kannada/
│       ├── train.csv               # Kannada-MNIST train set
│       └── test.csv                # Kannada-MNIST test set
│
├── models/                         # Local model checkpoints
│   ├── router.pt
│   ├── devanagari_model.pt
│   ├── best_bengali_cnn.pt
│   ├── tamil_model.pth
│   └── kannada_mnist_model.pt
│
├── streamlit-app/
│   ├── app.py                      # Streamlit application (two-stage inference)
│   ├── requirements.txt            # Python dependencies
│   ├── packages.txt                # System packages for HF Spaces
│   └── README.md                   # HF Spaces metadata + app docs
│
├── test_accuracy.py                # End-to-end accuracy evaluation script
│
├── bengali.ipynb                   # Bengali model training notebook
├── devanagiri.ipynb                # Devanagari model training notebook
├── tamil-dataset.ipynb             # Tamil model training notebook
├── kannada_mnist_recognition.ipynb # Kannada model training notebook
├── router-script-combined.ipynb    # Router CNN training notebook
│
└── README.md                       # This file
```

---

## Setup & Installation

### Prerequisites
- Python ≥ 3.9
- PyTorch ≥ 2.0 (CPU is fine; CUDA optional)

### Install dependencies

```bash
pip install torch torchvision pillow numpy huggingface_hub streamlit streamlit-drawable-canvas
```

Or using the app's requirements file:

```bash
pip install -r streamlit-app/requirements.txt
```

---

## Running the Streamlit App

```bash
cd streamlit-app
streamlit run app.py
```

The app downloads model weights from Hugging Face Hub on first launch (`kgan31/smail-A3`).

### App Modes

| Mode             | Description                                                      |
|------------------|------------------------------------------------------------------|
| ✏️ Draw & Predict | Draw a character on the canvas; get the predicted Unicode glyph  |
| 📁 Upload         | Upload a handwritten image for classification                    |
| 🎯 Practice Mode  | A target character is shown; draw it and see if the model agrees |

---


## Deployment

The Streamlit app is deployed on **Hugging Face Spaces**:

1. Push the `streamlit-app/` directory to a Hugging Face Space repository (SDK: Streamlit).
2. Model weights are stored in a separate model repo (`kgan31/smail-A3`) and fetched at runtime via `huggingface_hub`.
3. Set the `HF_REPO_ID` environment variable if using a different model repository.

```bash
# Override the model repo at runtime
HF_REPO_ID=your-username/your-model-repo streamlit run app.py
```

---

## Dependencies

| Package                   | Purpose                              |
|---------------------------|--------------------------------------|
| `torch` / `torchvision`   | Model inference                      |
| `Pillow`                  | Image loading and preprocessing      |
| `numpy`                   | Array operations                     |
| `streamlit`               | Web UI                               |
| `streamlit-drawable-canvas` | In-browser drawing canvas          |
| `huggingface_hub`         | Downloading model weights from HF Hub |

---

