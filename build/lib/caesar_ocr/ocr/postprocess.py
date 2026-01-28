"""OCR post-processing utilities."""

from __future__ import annotations

import cv2
import numpy as np
from PIL import Image

_FUER_VARIANTS = {"fiir", "fir", "flir", "filr", "flr", "fllr"}


def normalize_text(text: str) -> str:
    """Normalize whitespace in OCR text."""
    return " ".join(text.split())


def normalize_tokens(tokens):
    """Normalize token text and ensure stable ordering keys exist."""
    out = []
    for tok in tokens:
        text = str(tok.get("text", "")).strip()
        if not text:
            continue
        lower = text.lower()
        if lower in _FUER_VARIANTS:
            text = "für"
        tok = dict(tok)
        tok["text"] = text
        out.append(tok)
    return out


def _to_cv(im: Image.Image) -> np.ndarray:
    """Convert PIL image to OpenCV BGR array."""
    return cv2.cvtColor(np.array(im), cv2.COLOR_RGB2BGR)


def _from_cv(arr: np.ndarray) -> Image.Image:
    """Convert OpenCV BGR array to PIL image."""
    return Image.fromarray(cv2.cvtColor(arr, cv2.COLOR_BGR2RGB))


def preprocess_image(im: Image.Image) -> np.ndarray:
    """Preprocess image to improve OCR quality.

    Returns a grayscale, denoised image array suitable for Tesseract.
    """
    # Convert to OpenCV for processing.
    im = _to_cv(im)

    # Convert to grayscale.
    gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)

    # Light denoising reduces speckle without destroying glyphs.
    denoised = cv2.fastNlMeansDenoising(gray, h=8)

    return denoised
