"""Tesseract OCR engine adapter."""

from __future__ import annotations

from typing import Dict, List, Tuple

import pytesseract


def ocr_tokens(preprocessed_im, *, lang: str = "eng+deu", psm: int = 6) -> Tuple[str, List[Dict[str, object]]]:
    """Run OCR and return full text plus token dicts with bboxes/confidence."""
    cfg = f"--oem 3 --psm {psm}"
    data = pytesseract.image_to_data(
        preprocessed_im, lang=lang, config=cfg, output_type=pytesseract.Output.DICT
    )

    tokens: List[Dict[str, object]] = []
    n = len(data.get("text", []))
    for i in range(n):
        text = (data["text"][i] or "").strip()
        if not text:
            continue
        left = int(data["left"][i])
        top = int(data["top"][i])
        width = int(data["width"][i])
        height = int(data["height"][i])
        conf = float(data.get("conf", [])[i]) if data.get("conf") is not None else None
        tokens.append(
            {
                "text": text,
                "bbox": [left, top, left + width, top + height],
                "conf": conf,
                "block_num": int(data.get("block_num", [0])[i]),
                "par_num": int(data.get("par_num", [0])[i]),
                "line_num": int(data.get("line_num", [0])[i]),
                "word_num": int(data.get("word_num", [0])[i]),
            }
        )

    tokens.sort(key=lambda t: (t["block_num"], t["par_num"], t["line_num"], t["word_num"]))

    text_parts = []
    cursor = 0
    for token in tokens:
        if text_parts:
            text_parts.append(" ")
            cursor += 1
        start = cursor
        text_parts.append(token["text"])
        cursor += len(token["text"])
        end = cursor
        token["start"] = start
        token["end"] = end

    full_text = "".join(text_parts)
    return full_text, tokens


def ocr_tokens_from_image(image, *, lang: str = "eng+deu", psm: int = 6) -> Tuple[str, List[Dict[str, object]]]:
    """Run OCR on a PIL image and return text plus token dicts."""
    cfg = f"--oem 3 --psm {psm}"
    data = pytesseract.image_to_data(
        image, lang=lang, config=cfg, output_type=pytesseract.Output.DICT
    )
    tokens: List[Dict[str, object]] = []
    n = len(data.get("text", []))
    for i in range(n):
        text = (data["text"][i] or "").strip()
        if not text:
            continue
        left = int(data["left"][i])
        top = int(data["top"][i])
        width = int(data["width"][i])
        height = int(data["height"][i])
        conf = float(data.get("conf", [])[i]) if data.get("conf") is not None else None
        tokens.append(
            {
                "text": text,
                "bbox": [left, top, left + width, top + height],
                "conf": conf,
                "block_num": int(data.get("block_num", [0])[i]),
                "par_num": int(data.get("par_num", [0])[i]),
                "line_num": int(data.get("line_num", [0])[i]),
                "word_num": int(data.get("word_num", [0])[i]),
            }
        )

    tokens.sort(key=lambda t: (t["block_num"], t["par_num"], t["line_num"], t["word_num"]))

    text_parts = []
    cursor = 0
    for token in tokens:
        if text_parts:
            text_parts.append(" ")
            cursor += 1
        start = cursor
        text_parts.append(token["text"])
        cursor += len(token["text"])
        end = cursor
        token["start"] = start
        token["end"] = end

    full_text = "".join(text_parts)
    return full_text, tokens


def ocr_predictions(preprocessed_im, *, lang: str = "eng+deu", psm: int = 11) -> List[str]:
    """Run word-level OCR and return a list of tokens (lowercased)."""
    full_text, tokens = ocr_tokens(preprocessed_im, lang=lang, psm=psm)
    return [t["text"].lower() for t in tokens]


def ocr_text(preprocessed_im, *, lang: str = "eng+deu", psm: int = 3) -> str:
    """Run OCR and return the full text (not tokenized)."""
    cfg = f"--oem 3 --psm {psm}"
    return pytesseract.image_to_string(preprocessed_im, lang=lang, config=cfg)
