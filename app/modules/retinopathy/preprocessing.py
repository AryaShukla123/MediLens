import os
from functools import lru_cache

import cv2
import numpy as np
import torch
import torch.nn as nn
from torchvision import models

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "model.pt")

IMG_SIZE = 224
SUPPORTED_ARCHITECTURE = "mobilenet_v3_small"
NUM_CLASSES = 5

CLASS_NAMES = ["No DR", "Mild", "Moderate", "Severe", "Proliferative"]

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


# ---------------------------------------------------------------------------
# Image preprocessing
# ---------------------------------------------------------------------------

def crop_image_from_gray(img: np.ndarray, tol: int = 7) -> np.ndarray:
    """Crop away the black border around the retina (img is an RGB array)."""
    # COLOR_BGR2GRAY on an RGB array looks like a mix-up, but it is exactly
    # what the training notebook did, and it only affects the threshold test
    # on near-black border pixels. Kept identical so crops match training.
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    mask = gray > tol
    if mask.sum() == 0:  # image too dark to crop safely
        return img
    rows, cols = mask.any(1), mask.any(0)
    if rows.sum() == 0 or cols.sum() == 0:
        return img
    channels = [img[:, :, i][np.ix_(rows, cols)] for i in range(3)]
    return np.stack(channels, axis=-1)


def ben_graham_enhance(img: np.ndarray, sigma_x: int = 10) -> np.ndarray:
    """Subtract a blurred copy of the image to boost local contrast."""
    return cv2.addWeighted(img, 4, cv2.GaussianBlur(img, (0, 0), sigma_x), -4, 128)


def preprocess_retina_array(img_rgb: np.ndarray, img_size: int = IMG_SIZE):
    """
    Full preprocessing for one RGB image array.

    Returns (model_input_rgb, display_rgb):
      - model_input_rgb : crop + resize + Ben Graham enhancement. This is what
                          the CNN was trained on.
      - display_rgb     : crop + resize only. Looks like a normal fundus photo,
                          so it is the right base image to show the user and to
                          lay the Grad-CAM heatmap over.
    Both are uint8, shape (img_size, img_size, 3).
    """
    cropped = crop_image_from_gray(img_rgb)
    display_rgb = cv2.resize(cropped, (img_size, img_size))
    model_input_rgb = ben_graham_enhance(display_rgb)
    return model_input_rgb, display_rgb


def preprocess_retina_image(img_path: str, img_size: int = IMG_SIZE) -> np.ndarray:
    """
    File-path version, used by train.py when it builds the image cache.
    Returns only the enhanced image (same output the notebook cached).
    """
    img_bgr = cv2.imread(img_path)
    if img_bgr is None:
        raise ValueError(f"Could not read image: {img_path}")
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    model_input_rgb, _ = preprocess_retina_array(img_rgb, img_size)
    return model_input_rgb


def decode_image_bytes(image_bytes: bytes) -> np.ndarray:
    """Decode uploaded file bytes (png/jpg) to an RGB array, or raise ValueError."""
    buffer = np.frombuffer(image_bytes, dtype=np.uint8)
    img_bgr = cv2.imdecode(buffer, cv2.IMREAD_COLOR)
    if img_bgr is None:
        raise ValueError("The uploaded file is not a readable image.")
    return cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)


def to_model_tensor(model_input_rgb: np.ndarray) -> torch.Tensor:
    """uint8 RGB (H, W, 3) -> normalised float tensor of shape (1, 3, H, W)."""
    arr = model_input_rgb.astype(np.float32) / 255.0
    arr = (arr - IMAGENET_MEAN) / IMAGENET_STD
    return torch.from_numpy(arr.transpose(2, 0, 1)).unsqueeze(0).float()


# ---------------------------------------------------------------------------
# Model building / loading
# ---------------------------------------------------------------------------

def build_model(num_classes: int = NUM_CLASSES, pretrained: bool = False,
                freeze_backbone: bool = False) -> nn.Module:
    """
    MobileNetV3-Small with a fresh num_classes head.

    pretrained=True downloads ImageNet weights (training only). For inference
    the weights come from model.pt, so pretrained stays False.
    freeze_backbone=True reproduces the notebook's transfer-learning setup:
    everything frozen except the last 3 feature blocks and the new head.
    """
    weights = models.MobileNet_V3_Small_Weights.IMAGENET1K_V1 if pretrained else None
    model = models.mobilenet_v3_small(weights=weights)

    if freeze_backbone:
        for param in model.parameters():
            param.requires_grad = False
        for param in model.features[-3:].parameters():
            param.requires_grad = True

    in_features = model.classifier[-1].in_features
    model.classifier[-1] = nn.Linear(in_features, num_classes)
    return model


@lru_cache(maxsize=1)
def load_model():
    """
    Load model.pt once and keep it in memory. Returns (model, meta) where
    meta carries img_size / class_names stored alongside the weights.
    """
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            f"{MODEL_PATH} not found. Train the model first "
            "(notebooks/retinopathy_training.ipynb or "
            "`python -m app.modules.retinopathy.train`)."
        )

    checkpoint = torch.load(MODEL_PATH, map_location="cpu")

    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
        meta = {
            "architecture": checkpoint.get("architecture", SUPPORTED_ARCHITECTURE),
            "num_classes": checkpoint.get("num_classes", NUM_CLASSES),
            "img_size": checkpoint.get("img_size", IMG_SIZE),
            "class_names": checkpoint.get("class_names", CLASS_NAMES),
        }
    else:  # a bare state_dict was saved
        state_dict = checkpoint
        meta = {
            "architecture": SUPPORTED_ARCHITECTURE,
            "num_classes": NUM_CLASSES,
            "img_size": IMG_SIZE,
            "class_names": CLASS_NAMES,
        }

    if meta["architecture"] != SUPPORTED_ARCHITECTURE:
        # The training notebook started on ResNet50 before switching to
        # MobileNetV3-Small for CPU speed, so an old label here is plausible.
        # The weights decide whether loading works, not the label.
        print(f"[retinopathy] checkpoint says architecture={meta['architecture']!r}; "
              f"loading as {SUPPORTED_ARCHITECTURE}.")

    model = build_model(num_classes=meta["num_classes"], pretrained=False)
    try:
        model.load_state_dict(state_dict)
    except RuntimeError as exc:
        raise RuntimeError(
            "model.pt does not match the MobileNetV3-Small architecture used "
            "by this module. Retrain with train.py (or the updated notebook)."
        ) from exc

    model.eval()

    # MobileNetV3 uses in-place Hardswish/Dropout. Grad-CAM keeps references to
    # intermediate activations, and in-place ops overwriting them is a known
    # source of hook errors. Turning it off costs nothing at inference size.
    for module in model.modules():
        if hasattr(module, "inplace"):
            module.inplace = False

    return model, meta


# ---------------------------------------------------------------------------
# Prediction
# ---------------------------------------------------------------------------

def predict_retinopathy(image_bytes: bytes) -> dict:
    """
    Run the full inference pipeline on an uploaded image.

    Returns:
      prediction     - int class index 0..4
      label          - class name ("No DR", "Mild", ...)
      confidence     - probability of the predicted class (0..1)
      probabilities  - {class name: probability} for all 5 classes
      is_positive    - True when any retinopathy is predicted (class >= 1)
      input_tensor   - (1, 3, H, W) tensor, reused by gradcam.py
      display_rgb    - uint8 RGB image to show / overlay the heatmap on
    """
    model, meta = load_model()

    img_rgb = decode_image_bytes(image_bytes)
    model_input_rgb, display_rgb = preprocess_retina_array(img_rgb, meta["img_size"])
    input_tensor = to_model_tensor(model_input_rgb)

    with torch.no_grad():
        probs = torch.softmax(model(input_tensor), dim=1)[0].numpy()

    prediction = int(probs.argmax())
    class_names = meta["class_names"]

    return {
        "prediction": prediction,
        "label": class_names[prediction],
        "confidence": round(float(probs[prediction]), 4),
        "probabilities": {name: round(float(p), 4) for name, p in zip(class_names, probs)},
        "is_positive": prediction >= 1,
        "input_tensor": input_tensor,
        "display_rgb": display_rgb,
    }
