"""LayoutLMv3 token classification inference helpers."""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import torch
from PIL import Image
from transformers import AutoProcessor, LayoutLMv3ForTokenClassification


def _normalize_box(box: List[int], width: int, height: int) -> List[int]:
    x0, y0, x1, y1 = box
    return [
        max(0, min(1000, int(1000 * x0 / width))),
        max(0, min(1000, int(1000 * y0 / height))),
        max(0, min(1000, int(1000 * x1 / width))),
        max(0, min(1000, int(1000 * y1 / height))),
    ]


def _load_labels(model_dir: str, model) -> Dict[int, str]:
    labels_path = f"{model_dir}/labels.json"
    try:
        import json
        with open(labels_path, "r", encoding="utf-8") as f:
            labels = json.load(f)
        return {idx: label for idx, label in enumerate(labels)}
    except Exception:
        return model.config.id2label


def infer_tokens(
    image: Image.Image,
    tokens: List[str],
    bboxes: List[List[int]],
    *,
    model_dir: str,
    max_length: int = 512,
) -> Tuple[List[str], Optional[List[float]]]:
    processor = AutoProcessor.from_pretrained(model_dir, apply_ocr=False)
    model = LayoutLMv3ForTokenClassification.from_pretrained(model_dir)
    model.eval()

    width, height = image.size
    norm_boxes = [_normalize_box(b, width, height) for b in bboxes]

    encoding = processor(
        images=image,
        text=tokens,
        boxes=norm_boxes,
        truncation=True,
        padding="max_length",
        max_length=max_length,
        return_tensors="pt",
    )
    with torch.no_grad():
        logits = model(**encoding).logits.squeeze(0)

    probs = torch.softmax(logits, dim=-1)
    pred_ids = logits.argmax(-1).tolist()
    id2label = _load_labels(model_dir, model)
    labels = [id2label.get(idx, "O") for idx in pred_ids[: len(tokens)]]
    scores = probs.max(dim=-1).values.tolist()[: len(tokens)]

    return labels, scores
