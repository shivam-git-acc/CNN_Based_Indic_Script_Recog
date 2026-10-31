"""
Multi-Script Handwritten Character Recognition — Streamlit App
==============================================================
Draw a character on the canvas (or upload an image) and get the Unicode prediction.
Supports Devanagari, Tamil, Bengali, and Kannada scripts.

Pipeline:
  1. Router CNN predicts the script (Devanagari / Tamil / Bengali / Kannada)
  2. Script-specific CNN predicts the character class
  3. Class label → Unicode mapping gives the final prediction

Models are fetched from Hugging Face Hub.
"""
from PIL import ImageFilter
import streamlit as st
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from streamlit_drawable_canvas import st_canvas
from huggingface_hub import hf_hub_download
import json
import os
import random

# ─────────────────────────────────────────────────────────────
# 1.  MODEL ARCHITECTURES  (must match training code exactly)
# ─────────────────────────────────────────────────────────────

# --- New Router CNN Architecture ---
class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch, kernel=3, stride=1, pool=True, residual=False):
        super().__init__()
        pad = kernel // 2
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel, stride=stride, padding=pad, bias=False),
            nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False), nn.BatchNorm2d(out_ch),
        )
        self.pool     = nn.MaxPool2d(2) if pool else nn.Identity()
        self.act      = nn.ReLU(inplace=True)
        self.residual = residual
        self.skip = (
            nn.Sequential(nn.Conv2d(in_ch, out_ch, 1, bias=False), nn.BatchNorm2d(out_ch))
            if residual and in_ch != out_ch else nn.Identity()
        )

    def forward(self, x):
        out = self.conv(x)
        if self.residual: out = out + self.skip(x)
        return self.pool(self.act(out))

class RouterCNN(nn.Module):
    def __init__(self, num_classes: int = 4, dropout: float = 0.4):
        super().__init__()
        self.features = nn.Sequential(
            # Block 1
            nn.Conv2d(1, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(True),
            nn.Conv2d(32, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(True),
            nn.MaxPool2d(2), nn.Dropout2d(0.1),           # 32 → 16
            # Block 2
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(True),
            nn.Conv2d(64, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(True),
            nn.MaxPool2d(2), nn.Dropout2d(0.2),           # 16 → 8
            # Block 3
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(True),
            nn.MaxPool2d(2),                              # 8  → 4
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 4 * 4, 256), nn.ReLU(True), nn.Dropout(dropout),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        return self.classifier(self.features(x))


# --- Script Specific CNN Architectures ---

class ModelA(nn.Module):
    def __init__(self, num_classes=46):
        super(ModelA, self).__init__()
        self.conv1 = nn.Conv2d(1, 64, kernel_size=5, stride=1, padding=2)
        self.bn1   = nn.BatchNorm2d(64)
        self.pool1 = nn.MaxPool2d(2, stride=2)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.bn2   = nn.BatchNorm2d(128)
        self.pool2 = nn.MaxPool2d(2, stride=2)
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1)
        self.bn3   = nn.BatchNorm2d(256)
        self.pool3 = nn.MaxPool2d(2, stride=2)
        self.dropout1 = nn.Dropout(0.5)
        self.fc1      = nn.Linear(256 * 4 * 4, 512)
        self.dropout2 = nn.Dropout(0.3)
        self.fc2      = nn.Linear(512, num_classes)

    def forward(self, x):
        x = self.pool1(F.relu(self.bn1(self.conv1(x))))
        x = self.pool2(F.relu(self.bn2(self.conv2(x))))
        x = self.pool3(F.relu(self.bn3(self.conv3(x))))
        x = x.view(x.size(0), -1)
        x = self.dropout1(F.relu(self.fc1(x)))
        x = self.dropout2(x)
        return self.fc2(x)
# class ModelA(nn.Module):
#     """Devanagari character classifier (46 classes, 32×32 input)."""
#     def __init__(self, num_classes=46):
#         super(ModelA, self).__init__()
#         self.conv1 = nn.Conv2d(1, 64, kernel_size=5, stride=1)
#         self.lrn1  = nn.LocalResponseNorm(5, alpha=0.001, beta=0.75, k=1.0)
#         self.pool1 = nn.MaxPool2d(2, stride=2)

#         self.conv2 = nn.Conv2d(64, 128, kernel_size=5, stride=1)
#         self.lrn2  = nn.LocalResponseNorm(5, alpha=0.001, beta=0.75, k=1.0)
#         self.pool2 = nn.MaxPool2d(2, stride=2)

#         self.conv3 = nn.Conv2d(128, 256, kernel_size=5, stride=1)
#         self.lrn3  = nn.LocalResponseNorm(5, alpha=0.001, beta=0.75, k=1.0)

#         self.dropout = nn.Dropout(0.5)
#         self.fc      = nn.Linear(256, num_classes)

#     def forward(self, x):
#         x = self.pool1(F.relu(self.lrn1(self.conv1(x))))
#         x = self.pool2(F.relu(self.lrn2(self.conv2(x))))
#         x = F.relu(self.lrn3(self.conv3(x)))
#         x = x.view(x.size(0), -1)
#         x = self.dropout(x)
#         return self.fc(x)

class BengaliCNN(nn.Module):
    """Bengali character classifier (84 classes, 32×32 input)."""
    def __init__(self, num_classes=84, dropout=0.4):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1, bias=False), nn.BatchNorm2d(32), nn.ReLU(True),
            nn.Conv2d(32, 32, 3, padding=1, bias=False), nn.BatchNorm2d(32), nn.ReLU(True),
            nn.MaxPool2d(2, 2), nn.Dropout2d(0.1),
            nn.Conv2d(32, 64, 3, padding=1, bias=False), nn.BatchNorm2d(64), nn.ReLU(True),
            nn.Conv2d(64, 64, 3, padding=1, bias=False), nn.BatchNorm2d(64), nn.ReLU(True),
            nn.MaxPool2d(2, 2), nn.Dropout2d(0.15),
            nn.Conv2d(64, 128, 3, padding=1, bias=False), nn.BatchNorm2d(128), nn.ReLU(True),
            nn.Conv2d(128, 128, 3, padding=1, bias=False), nn.BatchNorm2d(128), nn.ReLU(True),
            nn.MaxPool2d(2, 2),
        )
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128, 256), nn.ReLU(True), nn.Dropout(dropout),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.pool(x)
        return self.classifier(x)

class TamilCNN(nn.Module):
    """Tamil character classifier (156 classes, 32×32 input)."""
    def __init__(self, num_classes=156, dropout=0.4):
        super().__init__()
        self.block1 = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(True),
            nn.Conv2d(32, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(True),
            nn.MaxPool2d(2, 2), nn.Dropout2d(0.1)
        )
        self.block2 = nn.Sequential(
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(True),
            nn.Conv2d(64, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(True),
            nn.MaxPool2d(2, 2), nn.Dropout2d(0.15)
        )
        self.block3 = nn.Sequential(
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(True),
            nn.Conv2d(128, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(True),
            nn.MaxPool2d(2, 2), nn.Dropout2d(0.2)
        )
        self.block4 = nn.Sequential(
            nn.Conv2d(128, 256, 3, padding=1), nn.BatchNorm2d(256), nn.ReLU(True),
            nn.Conv2d(256, 256, 3, padding=1), nn.BatchNorm2d(256), nn.ReLU(True),
            nn.MaxPool2d(2, 2), nn.Dropout2d(0.25)
        )
        self.pool = nn.AdaptiveAvgPool2d(1)
        
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256, 512), nn.ReLU(True), nn.Dropout(dropout),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.pool(x)
        return self.classifier(x)

class KannadaCNN(nn.Module):
    """Kannada digit classifier (10 classes, 28×28 input)."""
    def __init__(self, num_classes=10):
        super().__init__()
        self.block1 = nn.Sequential(
            nn.Conv2d(1, 32, 5, padding=2), nn.ReLU(),
            nn.Conv2d(32, 32, 5, padding=2), nn.ReLU(),
            nn.BatchNorm2d(32, momentum=0.15),
            nn.MaxPool2d(2, 2), nn.Dropout2d(0.25),
        )
        self.block2 = nn.Sequential(
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1), nn.ReLU(),
            nn.BatchNorm2d(64, momentum=0.15),
            nn.MaxPool2d(2, 2), nn.Dropout2d(0.25),
        )
        self.block3 = nn.Sequential(
            nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(),
            nn.BatchNorm2d(128, momentum=0.15), nn.Dropout2d(0.25),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 7 * 7, 256), nn.ReLU(), nn.Dropout(0.4),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        return self.classifier(x)


# ─────────────────────────────────────────────────────────────
# 2.  CONFIGURATION & LABEL MAPS
# ─────────────────────────────────────────────────────────────

DEVICE = torch.device("cpu")

# Hugging Face repo where your models are hosted
HF_REPO_ID = os.environ.get("HF_REPO_ID", "kgan31/smail-A3")

SCRIPT_NAMES = ["devanagari", "tamil", "bengali", "kannada"]
SCRIPT_TO_IDX = {s: i for i, s in enumerate(SCRIPT_NAMES)}

# Unicode mappings for each script
DEVANAGARI_LABELS = {
    "character_1_ka": "क", "character_2_kha": "ख", "character_3_ga": "ग",
    "character_4_gha": "घ", "character_5_kna": "ङ", "character_6_cha": "च",
    "character_7_chha": "छ", "character_8_ja": "ज", "character_9_jha": "झ",
    "character_10_yna": "ञ", "character_11_taamatar": "ट",
    "character_12_thaa": "ठ", "character_13_daa": "ड", "character_14_dhaa": "ढ",
    "character_15_adna": "ण", "character_16_tabala": "त", "character_17_tha": "थ",
    "character_18_da": "द", "character_19_dha": "ध", "character_20_na": "न",
    "character_21_pa": "प", "character_22_pha": "फ", "character_23_ba": "ब",
    "character_24_bha": "भ", "character_25_ma": "म", "character_26_yaw": "य",
    "character_27_ra": "र", "character_28_la": "ल", "character_29_waw": "व",
    "character_30_motosaw": "श", "character_31_petchiryakha": "ष",
    "character_32_patalosaw": "स", "character_33_ha": "ह",
    "character_34_chhya": "क्ष", "character_35_tra": "त्र",
    "character_36_gya": "ज्ञ",
    "digit_0": "०", "digit_1": "१", "digit_2": "२", "digit_3": "३",
    "digit_4": "४", "digit_5": "५", "digit_6": "६", "digit_7": "७",
    "digit_8": "८", "digit_9": "९",
}

KANNADA_LABELS = {
    f"class_{i}": chr(0x0CE6 + i) for i in range(10)  # Kannada digits ೦-೯
}

# For Bengali and Tamil, we use generic class labels
# These would be populated from the model checkpoint or a mapping file
BENGALI_LABELS = {f"class_{i+1:02d}": f"Bengali Class {i+1}" for i in range(84)}
TAMIL_LABELS = {f"class_{i+1:03d}": f"Tamil Class {i+1}" for i in range(156)}


# ─────────────────────────────────────────────────────────────
# 3.  MODEL LOADING (cached)
# ─────────────────────────────────────────────────────────────

@st.cache_resource
def load_models():
    """Download & load all five models from Hugging Face Hub."""
    models = {}

    # Router
    try:
        router_path = hf_hub_download(repo_id=HF_REPO_ID, filename="router.pt")
        ckpt = torch.load(router_path, map_location=DEVICE, weights_only=False)
        router = RouterCNN(num_classes=4)
        
        if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
            router.load_state_dict(ckpt["model_state_dict"])
        else:
            router.load_state_dict(ckpt)
            
        router.eval()
        models["router"] = router
    except Exception as e:
        st.error(f"Failed to load router model: {e}")
        models["router"] = None

    # Devanagari
    # try:
    #     dev_path = hf_hub_download(repo_id=HF_REPO_ID, filename="devanagari_model.pt")
    #     ckpt = torch.load(dev_path, map_location=DEVICE, weights_only=False)
        
    #     if isinstance(ckpt, dict):
    #         dev_model = ModelA(num_classes=ckpt.get("num_classes", 46))
    #         if "model_state" in ckpt:
    #             dev_model.load_state_dict(ckpt["model_state"])
    #         else:
    #             dev_model.load_state_dict(ckpt)
    #     else:
    #         # If it's a full model object, extract its state_dict
    #         dev_model = ModelA(num_classes=46)
    #         dev_model.load_state_dict(ckpt.state_dict())
            
    #     dev_model.eval()
    #     models["devanagari"] = dev_model
    # except Exception as e:
    #     st.warning(f"Devanagari model not loaded: {e}")
    #     models["devanagari"] = None

    try:
        # 1. Download the file
        dev_path = hf_hub_download(repo_id=HF_REPO_ID, filename="devanagari_model2.pt")
        
        # 2. Load the checkpoint
        ckpt = torch.load(dev_path, map_location=DEVICE, weights_only=False)
        
        # 3. Initialize the model and load the exact state dictionary (Your simplified code)
        dev_model = ModelA(num_classes=ckpt.get("num_classes", 46))
        dev_model.load_state_dict(ckpt["model_state"])
        dev_model.eval()
        
        # 4. Store it
        models["devanagari"] = dev_model
        
    except Exception as e:
        # Fallback if anything goes wrong (download fails, wrong dict keys, etc.)
        st.warning(f"Devanagari model not loaded: {e}")
        models["devanagari"] = None

    # Bengali
    try:
        ben_path = hf_hub_download(repo_id=HF_REPO_ID, filename="bengali_cnn_final.pt")
        ckpt = torch.load(ben_path, map_location=DEVICE, weights_only=False)
        
        if isinstance(ckpt, dict):
            ben_model = BengaliCNN(num_classes=ckpt.get("num_classes", 84))
            if "model_state" in ckpt:
                ben_model.load_state_dict(ckpt["model_state"])
            else:
                ben_model.load_state_dict(ckpt)
        else:
            ben_model = BengaliCNN(num_classes=84)
            ben_model.load_state_dict(ckpt.state_dict())
            
        ben_model.eval()
        models["bengali"] = ben_model
    except Exception as e:
        st.warning(f"Bengali model not loaded: {e}")
        models["bengali"] = None

    # Tamil
    try:
        tam_path = hf_hub_download(repo_id=HF_REPO_ID, filename="tamil_model.pth")
        ckpt = torch.load(tam_path, map_location=DEVICE, weights_only=False)
        
        if isinstance(ckpt, dict):
            tam_model = TamilCNN(num_classes=ckpt.get("num_classes", 156))
            # Support multiple possible save keys for the state dictionary
            if "model_state_dict" in ckpt:
                tam_model.load_state_dict(ckpt["model_state_dict"])
            elif "model_state" in ckpt:
                tam_model.load_state_dict(ckpt["model_state"])
            else:
                tam_model.load_state_dict(ckpt)
        else:
            tam_model = TamilCNN(num_classes=156)
            tam_model.load_state_dict(ckpt.state_dict())
            
        tam_model.eval()
        models["tamil"] = tam_model
    except Exception as e:
        st.warning(f"Tamil model not loaded: {e}")
        models["tamil"] = None

    # Kannada
    try:
        kan_path = hf_hub_download(repo_id=HF_REPO_ID, filename="kannada_mnist_model.pt")
        ckpt = torch.load(kan_path, map_location=DEVICE, weights_only=False)
        
        if isinstance(ckpt, dict):
            kan_model = KannadaCNN(num_classes=ckpt.get("num_classes", 10))
            if "model_state" in ckpt:
                kan_model.load_state_dict(ckpt["model_state"])
            else:
                kan_model.load_state_dict(ckpt)
        else:
            kan_model = KannadaCNN(num_classes=10)
            kan_model.load_state_dict(ckpt.state_dict())
            
        kan_model.eval()
        models["kannada"] = kan_model
    except Exception as e:
        st.warning(f"Kannada model not loaded: {e}")
        models["kannada"] = None

    return models


# ─────────────────────────────────────────────────────────────
# 4.  PREPROCESSING
# ─────────────────────────────────────────────────────────────

def preprocess_image(image: Image.Image, target_size: int = 32) -> torch.Tensor:
    """Convert a PIL image to a normalised grayscale tensor."""
    img = image.convert("L")
    img = img.resize((target_size, target_size), Image.BILINEAR)
    arr = np.array(img, dtype=np.float32) / 255.0
    tensor = torch.from_numpy(arr).unsqueeze(0).unsqueeze(0)  # (1,1,H,W)
    tensor = (tensor - 0.5) / 0.5  # normalise to [-1, 1]
    return tensor


# ─────────────────────────────────────────────────────────────
# 5.  PREDICTION PIPELINE
# ─────────────────────────────────────────────────────────────

def predict(image: Image.Image, models: dict):
    """Run the two-stage prediction pipeline."""
    router = models.get("router")
    if router is None:
        return None, None, None, None

    # Stage 1: Identify the script
    x_router = preprocess_image(image, target_size=32)
    with torch.no_grad():
        logits = router(x_router)
        probs = F.softmax(logits, dim=1)[0]
        script_idx = probs.argmax().item()
        script_name = SCRIPT_NAMES[script_idx]
        script_conf = probs[script_idx].item()

    # Stage 2: Identify the character
    char_model = models.get(script_name)
    if char_model is None:
        return script_name, script_conf, None, None

    # Kannada uses 28×28 input
    target_size = 28 if script_name == "kannada" else 32
    x_char = preprocess_image(image, target_size=target_size)

    with torch.no_grad():
        logits = char_model(x_char)
        probs = F.softmax(logits, dim=1)[0]
        char_idx = probs.argmax().item()
        char_conf = probs[char_idx].item()

    # Map class index to label/unicode
    label_map = {
        "devanagari": DEVANAGARI_LABELS,
        "bengali": BENGALI_LABELS,
        "tamil": TAMIL_LABELS,
        "kannada": KANNADA_LABELS,
    }

    # Get sorted class names (matching how ImageFolder sorts them)
    script_labels = label_map.get(script_name, {})
    sorted_keys = sorted(script_labels.keys())
    if char_idx < len(sorted_keys):
        class_name = sorted_keys[char_idx]
        unicode_char = script_labels[class_name]
    else:
        class_name = f"class_{char_idx}"
        unicode_char = "?"

    return script_name, script_conf, (class_name, unicode_char), char_conf


# ─────────────────────────────────────────────────────────────
# 6.  PRACTICE MODE TARGETS
# ─────────────────────────────────────────────────────────────

PRACTICE_TARGETS = {
    "devanagari": list(DEVANAGARI_LABELS.values()),
    "kannada": list(KANNADA_LABELS.values()),
    "bengali": ["ক", "খ", "গ", "ঘ", "ঙ", "চ", "ছ", "জ", "ঝ", "ঞ"],
    "tamil": ["அ", "ஆ", "இ", "ஈ", "உ", "ஊ", "எ", "ஏ", "ஐ", "ஒ"],
}


# ─────────────────────────────────────────────────────────────
# 7.  STREAMLIT UI
# ─────────────────────────────────────────────────────────────

def main():
    st.set_page_config(
        page_title="Multi-Script Character Recognition",
        page_icon="✍️",
        layout="wide",
    )

    st.title("✍️ Multi-Script Handwritten Character Recognition")
    st.markdown("""
    **Draw** a character on the canvas or **upload** an image.
    The system will:
    1. **Identify the script** (Devanagari, Tamil, Bengali, or Kannada)
    2. **Predict the character** and show its Unicode representation
    """)

    # Load models
    with st.spinner("Loading models from Hugging Face Hub..."):
        models = load_models()

    # Sidebar — mode selector
    mode = st.sidebar.radio("Mode", ["✏️ Draw & Predict", "🎯 Practice Mode"])

    if mode == "✏️ Draw & Predict":
        draw_and_predict_mode(models)
    else:
        practice_mode(models)


def draw_and_predict_mode(models):
    """Main draw-and-predict interface."""
    col1, col2 = st.columns([1, 1])

    with col1:
        st.subheader("🖊️ Draw or Upload")

        input_method = st.radio("Input method", ["Draw", "Upload"], horizontal=True)

        if input_method == "Draw":
            canvas_result = st_canvas(
                fill_color="rgba(0, 0, 0, 1)",
                stroke_width=9,
                stroke_color="#000000",
                background_color="#FFFFFF",
                width=280,
                height=280,
                drawing_mode="freedraw",
                key="canvas_draw",
            )
            image = None
            if canvas_result.image_data is not None:
                img_arr = canvas_result.image_data[:, :, :3]
                if np.any(img_arr < 255):  # something was drawn (dark pixels exist)
                    image = Image.fromarray(img_arr.astype(np.uint8)).convert("L")

                    # Crop tightly to drawn strokes
                    arr = np.array(image)
                    dark_pixels = arr < 200
                    rows = np.any(dark_pixels, axis=1)
                    cols = np.any(dark_pixels, axis=0)
                    if rows.any() and cols.any():
                        rmin, rmax = np.where(rows)[0][[0, -1]]
                        cmin, cmax = np.where(cols)[0][[0, -1]]
                        pad = max(10, int(0.1 * max(rmax - rmin, cmax - cmin)))
                        rmin = max(0, rmin - pad)
                        rmax = min(arr.shape[0], rmax + pad)
                        cmin = max(0, cmin - pad)
                        cmax = min(arr.shape[1], cmax + pad)
                        image = image.crop((cmin, rmin, cmax, rmax))

                    # Blur to match scanned image texture
                    image = image.filter(ImageFilter.GaussianBlur(radius=2))

        else:
            uploaded = st.file_uploader("Upload an image", type=["png", "jpg", "jpeg", "bmp"])
            image = None
            if uploaded is not None:
                image = Image.open(uploaded).convert("L")
                st.image(image, caption="Uploaded Image", width=280)

    with col2:
        st.subheader("🔮 Prediction")

        if image is not None:
            script_name, script_conf, char_info, char_conf = predict(image, models)

            if script_name is not None:
                st.success(f"**Script detected:** {script_name.capitalize()} ({script_conf*100:.1f}%)")

                if char_info is not None:
                    class_name, unicode_char = char_info
                    st.markdown(f"""
                    ### Predicted Character
                    | Property | Value |
                    |----------|-------|
                    | **Unicode** | `{unicode_char}` |
                    | **Class** | `{class_name}` |
                    | **Confidence** | {char_conf*100:.1f}% |
                    """)
                    st.markdown(
                        f'<div style="font-size:120px; text-align:center; padding:20px; '
                        f'background:#1a1a2e; border-radius:10px; color:#e94560;">'
                        f'{unicode_char}</div>',
                        unsafe_allow_html=True,
                    )
                else:
                    st.warning("Character model not available for this script.")
            else:
                st.error("Router model not loaded. Cannot make predictions.")
        else:
            st.info("Draw a character or upload an image to get a prediction.")

def practice_mode(models):
    """Practice mode: shows a target character, user draws it, and gets feedback."""
    st.subheader("🎯 Practice Mode")
    st.markdown("Draw the **target character** and see if the model recognises it!")

    col1, col2, col3 = st.columns([1, 1, 1])

    with col1:
        script = st.selectbox("Choose a script", list(PRACTICE_TARGETS.keys()))
        targets = PRACTICE_TARGETS[script]

        if st.button("🎲 New Target"):
            st.session_state["practice_target"] = random.choice(targets)

        if "practice_target" not in st.session_state:
            st.session_state["practice_target"] = random.choice(targets)

        target_char = st.session_state["practice_target"]

        st.markdown("### Target Character:")
        st.markdown(
            f'<div style="font-size:100px; text-align:center; padding:20px; '
            f'background:#0f3460; border-radius:10px; color:#e94560;">'
            f'{target_char}</div>',
            unsafe_allow_html=True,
        )

    with col2:
        st.markdown("### Your Drawing:")
        canvas_result = st_canvas(
            fill_color="rgba(0, 0, 0, 1)",
            stroke_width=12,
            stroke_color="#FFFFFF",
            background_color="#000000",
            width=280,
            height=280,
            drawing_mode="freedraw",
            key="canvas_practice",
        )

    with col3:
        st.markdown("### Feedback:")

        image = None
        if canvas_result.image_data is not None:
            img_arr = canvas_result.image_data[:, :, :3]
            if np.any(img_arr > 0):
                image = Image.fromarray(img_arr.astype(np.uint8)).convert("L")

        if image is not None:
            script_name, script_conf, char_info, char_conf = predict(image, models)

            if char_info is not None:
                _, predicted_char = char_info
                if predicted_char == target_char:
                    st.success(f"✅ **Correct!** The model recognised `{predicted_char}` ({char_conf*100:.1f}%)")
                    st.balloons()
                else:
                    st.error(
                        f"❌ **Not quite.** Expected `{target_char}`, "
                        f"got `{predicted_char}` ({char_conf*100:.1f}%)"
                    )
                    st.info("Try drawing more clearly, or click 'New Target' to try another character.")
            elif script_name:
                st.warning(f"Detected script: {script_name}, but character model unavailable.")
            else:
                st.error("Models not loaded.")
        else:
            st.info("Draw the target character on the canvas.")


if __name__ == "__main__":
    main()