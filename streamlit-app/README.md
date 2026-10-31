---
title: Multi-Script Handwritten Character Recognition
emoji: ✍️
colorFrom: blue
colorTo: purple
sdk: streamlit
sdk_version: "1.28.0"
app_file: app.py
pinned: false
---
# Multi-Script Handwritten Character Recognition

A **Streamlit** application that recognises handwritten characters from four Indian scripts:
**Devanagari**, **Tamil**, **Bengali**, and **Kannada**.

## How It Works

The system uses a **two-stage CNN pipeline**:

1. **Router CNN** – Identifies the script of the input image (4-class classifier)
2. **Script-specific CNN** – Predicts the exact character within that script

Models are hosted on [Hugging Face Hub](https://huggingface.co/) and downloaded automatically on first launch.

## Features

- **Draw & Predict** – Draw a character on the interactive canvas and get instant predictions
- **Upload** – Upload a handwritten character image for classification
- **Practice Mode** – A target character is shown; draw it and get real-time feedback

## Running Locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Model Architecture

| Model       | Input Size | Classes | Architecture            |
|-------------|-----------|---------|-------------------------|
| Router      | 32×32     | 4       | RouterCNN (3 conv blocks) |
| Devanagari  | 32×32     | 46      | DevanagariCNN            |
| Bengali     | 32×32     | 84      | BengaliCNN + AdaptivePool|
| Tamil       | 32×32     | 156     | TamilCNN + AdaptivePool  |
| Kannada     | 28×28     | 10      | KannadaCNN               |

## Environment Variables

| Variable      | Default                  | Description                    |
|---------------|--------------------------|--------------------------------|
| `HF_REPO_ID`  | `kgan31/smai-a3-models`  | Hugging Face repo with models  |

## Project Structure

```
streamlit-app/
├── app.py              # Main Streamlit application
├── requirements.txt    # Python dependencies
├── README.md           # This file
└── packages.txt        # System-level deps (for HF Spaces)
```

## Deployment (Hugging Face Spaces)

1. Create a new Space on Hugging Face (type: **Streamlit**)
2. Push this directory's contents to the Space repo
3. Set the `HF_REPO_ID` environment variable if using a different model repo
4. The app will start automatically

## Credits

Built as part of the SMAI Assignment 3 project at IIIT Hyderabad.
