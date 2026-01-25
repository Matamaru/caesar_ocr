"""OCR post-processing utilities."""

from __future__ import annotations

import cv2
import numpy as np
from PIL import Image


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
