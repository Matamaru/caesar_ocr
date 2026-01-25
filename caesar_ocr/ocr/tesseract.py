"""Tesseract OCR engine adapter."""

from __future__ import annotations

from typing import List

import pytesseract


def ocr_predictions(preprocessed_im, *, lang: str = "eng+deu", psm: int = 11) -> List[str]:
    """Run word-level OCR and return a list of tokens (lowercased)."""
    cfg = f"--oem 3 --psm {psm}"
    df = pytesseract.image_to_data(
        preprocessed_im, lang=lang, config=cfg, output_type=pytesseract.Output.DATAFRAME
    )
    df = df.dropna(subset=["text"])
    df = df[df["text"].str.strip() != ""]
    df = df.reset_index(drop=True)
    df = df[df["conf"] >= 0]
    df["text"] = df["text"].str.lower()
    return df["text"].to_list()


def ocr_text(preprocessed_im, *, lang: str = "eng+deu", psm: int = 3) -> str:
    """Run OCR and return the full text (not tokenized)."""
    cfg = f"--oem 3 --psm {psm}"
    return pytesseract.image_to_string(preprocessed_im, lang=lang, config=cfg)
