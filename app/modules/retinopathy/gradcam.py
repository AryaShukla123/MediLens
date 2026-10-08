"""
Grad-CAM for the retinopathy model.

Grad-CAM answers "which parts of the retina photo pushed the model toward
this grade?". It takes the activations of the model's last convolutional
block, weights them by how strongly each one increased the predicted class's
score, and turns that into a 0..1 heat map the same size as the input.

Public API (what routes.py will call):

    gc = generate_gradcam(pred["input_tensor"], pred["display_rgb"], pred["prediction"])
    gc["urls"]       -> {"original", "heatmap", "overlay"} web paths under /static/gradcam/
    gc["attention"]  -> plain-language description of where the model looked
                        (used later by the chatbot's context_builder)

Three images are saved per prediction so the page can offer an opacity
slider (gradcam_viewer.js) without recomputing anything:
  - original : the cropped retina photo
  - heatmap  : RGBA colour map, transparent where activation is low, meant to
               be layered on top of the original with CSS opacity
  - overlay  : the two already blended (used as the stored/thumbnail image)
"""

import os
import threading
import uuid

import cv2
import numpy as np
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

from app.modules.retinopathy.preprocessing import load_model

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
GRADCAM_DIR = os.path.join(BASE_DIR, "..", "..", "static", "gradcam")
GRADCAM_URL_PREFIX = "/static/gradcam"

# The model instance from load_model() is shared by every request and
# Grad-CAM attaches hooks to it, so only one heat map is computed at a time.
_GRADCAM_LOCK = threading.Lock()


# ---------------------------------------------------------------------------
# Core computation
# ---------------------------------------------------------------------------

def compute_gradcam(input_tensor, class_idx: int) -> np.ndarray:
    """
    Returns a float32 array of shape (H, W) with values in 0..1 for the given
    class. Targets features[-1], the last convolutional block of
    MobileNetV3-Small (7x7 feature map for a 224px input, upsampled to 224).
    """
    model, _ = load_model()
    target_layers = [model.features[-1]]

    with _GRADCAM_LOCK:
        with GradCAM(model=model, target_layers=target_layers) as cam:
            grayscale_cam = cam(
                input_tensor=input_tensor,
                targets=[ClassifierOutputTarget(int(class_idx))],
            )
    return grayscale_cam[0].astype(np.float32)


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------

def colorize_heatmap(cam: np.ndarray) -> np.ndarray:
    """0..1 cam -> uint8 RGB JET colour map."""
    heat_bgr = cv2.applyColorMap(np.uint8(255 * np.clip(cam, 0, 1)), cv2.COLORMAP_JET)
    return cv2.cvtColor(heat_bgr, cv2.COLOR_BGR2RGB)


def blend_overlay(display_rgb: np.ndarray, cam: np.ndarray, max_strength: float = 0.6) -> np.ndarray:
    """
    Blend the colour map onto the photo, strongest where activation is high
    and leaving low-activation areas as the untouched photo.
    """
    heat_rgb = colorize_heatmap(cam).astype(np.float32)
    weight = (np.clip(cam, 0, 1) * max_strength)[:, :, None]
    blended = display_rgb.astype(np.float32) * (1 - weight) + heat_rgb * weight
    return np.clip(blended, 0, 255).astype(np.uint8)


def heatmap_rgba(cam: np.ndarray) -> np.ndarray:
    """uint8 RGBA heat map whose alpha follows the activation strength."""
    rgb = colorize_heatmap(cam)
    alpha = np.uint8(255 * np.clip(cam, 0, 1))
    return np.dstack([rgb, alpha])


# ---------------------------------------------------------------------------
# Plain-language description (for the chatbot context later)
# ---------------------------------------------------------------------------

def describe_attention(cam: np.ndarray, threshold: float = 0.6) -> dict:
    """
    Summarise where the heat map is concentrated. Left/right/upper/lower refer
    to the image as displayed on screen, not to the patient's anatomy.
    """
    h, w = cam.shape
    mask = cam >= threshold
    if not mask.any():
        return {
            "region": "no single area",
            "spread": "diffuse",
            "area_fraction": 0.0,
            "summary": "The model's attention was spread thinly across the image "
                       "with no single standout area.",
        }

    weights = cam * mask
    total = float(weights.sum())
    ys, xs = np.indices(cam.shape)
    cy = float((weights * ys).sum()) / total / h   # 0 = top,  1 = bottom
    cx = float((weights * xs).sum()) / total / w   # 0 = left, 1 = right

    vertical = "upper" if cy < 1 / 3 else "lower" if cy > 2 / 3 else "middle"
    horizontal = "left" if cx < 1 / 3 else "right" if cx > 2 / 3 else "center"

    if vertical == "middle" and horizontal == "center":
        region = "central area"
    elif vertical == "middle":
        region = f"{horizontal} side"
    elif horizontal == "center":
        region = f"{vertical} part"
    else:
        region = f"{vertical}-{horizontal} area"

    area_fraction = float(mask.mean())
    spread = (
        "tightly focused" if area_fraction < 0.12
        else "moderately spread" if area_fraction < 0.30
        else "widely spread"
    )

    return {
        "region": region,
        "spread": spread,
        "area_fraction": round(area_fraction, 3),
        "summary": (
            f"The model's attention was {spread}, centred on the {region} of the "
            f"image (about {round(area_fraction * 100)}% of the image area)."
        ),
    }


# ---------------------------------------------------------------------------
# Saving + the one function routes.py needs
# ---------------------------------------------------------------------------

def save_gradcam_images(display_rgb: np.ndarray, cam: np.ndarray, out_dir: str = GRADCAM_DIR) -> dict:
    """Write original / heatmap / overlay PNGs and return their web URLs."""
    out_dir = os.path.abspath(out_dir)
    os.makedirs(out_dir, exist_ok=True)
    stem = uuid.uuid4().hex

    files = {
        "original": (f"{stem}_original.png", cv2.cvtColor(display_rgb, cv2.COLOR_RGB2BGR)),
        "heatmap": (f"{stem}_heatmap.png", cv2.cvtColor(heatmap_rgba(cam), cv2.COLOR_RGBA2BGRA)),
        "overlay": (f"{stem}_overlay.png", cv2.cvtColor(blend_overlay(display_rgb, cam), cv2.COLOR_RGB2BGR)),
    }

    urls = {}
    for key, (filename, image) in files.items():
        if not cv2.imwrite(os.path.join(out_dir, filename), image):
            raise IOError(f"Could not write Grad-CAM image to {out_dir}")
        urls[key] = f"{GRADCAM_URL_PREFIX}/{filename}"
    return urls


def generate_gradcam(input_tensor, display_rgb: np.ndarray, class_idx: int,
                     out_dir: str = GRADCAM_DIR) -> dict:
    """Compute the heat map for class_idx, save the images, describe the focus."""
    cam = compute_gradcam(input_tensor, class_idx)
    return {
        "cam": cam,
        "urls": save_gradcam_images(display_rgb, cam, out_dir),
        "attention": describe_attention(cam),
    }
