"""
utils.py — Shared utilities for the Porci-Integral AI module.

Consolidates duplicated code across app_fastapi.py, entrenar_v2.py,
cctv_monitor.py, and training_manager.py.
"""

import os
import json
import base64
import logging
from pathlib import Path
from io import BytesIO
from typing import Optional

import numpy as np
from PIL import Image


BASE_DIR = Path(__file__).resolve().parent
IMG_SIZE = (224, 224)
DEFAULT_THRESHOLD = 0.50
NEGATIVE_CLASS = "no_cerdo"  # folder name for negative/non-pig samples


def load_env(env_path: Optional[Path] = None):
    """Load .env file into os.environ if not already set.
    Idempotent — does not override existing environment variables.
    """
    env_path = env_path or BASE_DIR / ".env"
    if not env_path.exists():
        return
    with open(env_path) as _f:
        for _line in _f:
            _line = _line.strip()
            if _line and not _line.startswith("#") and "=" in _line:
                _k, _v = _line.split("=", 1)
                if _k.strip() not in os.environ:
                    os.environ[_k.strip()] = _v.strip().strip('"').strip("'")


def load_classes(classes_path: Optional[Path] = None) -> list:
    """Load class names from classes.json (authoritative).
    Falls back to scanning DATASET_PATH directories.
    """
    classes_path = classes_path or BASE_DIR / "classes.json"
    if classes_path.exists():
        with open(classes_path) as f:
            data = json.load(f)
        return data.get("classes", [])
    dataset_path = os.environ.get("DATASET_PATH", "")
    if dataset_path and os.path.exists(dataset_path):
        return sorted([
            d for d in os.listdir(dataset_path)
            if os.path.isdir(os.path.join(dataset_path, d))
        ])
    return []


def load_threshold(classes_path: Optional[Path] = None) -> float:
    """Load threshold from classes.json or return default."""
    classes_path = classes_path or BASE_DIR / "classes.json"
    if classes_path.exists():
        with open(classes_path) as f:
            data = json.load(f)
        return data.get("threshold", DEFAULT_THRESHOLD)
    return DEFAULT_THRESHOLD


def preprocess_image(image_bytes: bytes, img_size=IMG_SIZE) -> np.ndarray:
    """Convert raw image bytes to a normalized numpy array for inference."""
    img = Image.open(BytesIO(image_bytes)).convert("RGB")
    img = img.resize(img_size, Image.Resampling.BILINEAR)
    img_array = np.array(img, dtype=np.float32) / 255.0
    return np.expand_dims(img_array, axis=0)


def preprocess_image_tta(image_bytes: bytes, img_size=IMG_SIZE) -> list:
    """Preprocess image with Test-Time Augmentation variations.
    Returns list of (augmented_array, label) for ensemble averaging.
    """
    img = Image.open(BytesIO(image_bytes)).convert("RGB")

    def _to_array(pil_img):
        arr = pil_img.resize(img_size, Image.Resampling.BILINEAR)
        arr = np.array(arr, dtype=np.float32) / 255.0
        return np.expand_dims(arr, axis=0)

    versions = [
        (img.copy(), "original"),
        (img.transpose(Image.FLIP_LEFT_RIGHT), "flip"),
    ]

    # Rotate +10 degrees
    rot_p = img.rotate(10, resample=Image.Resampling.BILINEAR, expand=False, fillcolor="black")
    versions.append((rot_p, "rot+10"))

    # Rotate -10 degrees
    rot_n = img.rotate(-10, resample=Image.Resampling.BILINEAR, expand=False, fillcolor="black")
    versions.append((rot_n, "rot-10"))

    return [(_to_array(v), label) for v, label in versions]


def predict_with_tta(model, image_bytes, class_names, threshold=0.20):
    """Run inference with TTA: average predictions over multiple augmentations."""
    tta_inputs = preprocess_image_tta(image_bytes)
    all_preds = []

    for aug_array, _label in tta_inputs:
        preds = model.predict(aug_array, verbose=0)[0]
        all_preds.append(preds)

    avg_preds = np.mean(all_preds, axis=0)
    top_idx = int(np.argmax(avg_preds))
    confidence = float(avg_preds[top_idx])

    if confidence < threshold:
        class_name = "Desconocido"
    elif class_names[top_idx].lower() == "no_cerdo":
        class_name = "Desconocido"
    else:
        class_name = class_names[top_idx]

    return {
        "class_name": class_name,
        "confidence": confidence,
        "confidence_pct": f"{confidence * 100:.1f}%",
        "class_id": top_idx,
    }


def decode_base64_image(image_data: str) -> bytes:
    """Decode a base64 image string, stripping any data URI prefix."""
    if "," in image_data:
        image_data = image_data.split(",")[1]
    return base64.b64decode(image_data)


def setup_logging(name: str = __name__, level=logging.INFO):
    """Configure standard logging."""
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )
    return logging.getLogger(name)
