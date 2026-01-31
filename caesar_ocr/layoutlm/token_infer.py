"""LayoutLMv3 token classification inference helpers."""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import Dict, List, Optional, Tuple

import torch
from PIL import Image
from transformers import AutoProcessor, LayoutLMv3ForTokenClassification

from .utils import normalize_box


def _load_labels(model_dir: str, model) -> Dict[int, str]:
    labels_path = f"{model_dir}/labels.json"
    try:
        import json
        with open(labels_path, "r", encoding="utf-8") as f:
            labels = json.load(f)
        return {idx: label for idx, label in enumerate(labels)}
    except Exception:
        return model.config.id2label


def _align_predictions(pred_ids: List[int], word_ids: List[Optional[int]], id2label: Dict[int, str]) -> List[str]:
    labels: List[str] = ["O"] * (max([i for i in word_ids if i is not None], default=-1) + 1)
    seen = set()
    for idx, word_id in enumerate(word_ids):
        if word_id is None or word_id in seen:
            continue
        seen.add(word_id)
        labels[word_id] = id2label.get(pred_ids[idx], "O")
    return labels


@dataclass
class TokenInferer:
    model_dir: str
    processor: object
    model: object
    id2label: Dict[int, str]

    @classmethod
    def from_model_dir(cls, model_dir: str) -> "TokenInferer":
        return _load_token_inferer(model_dir)

    def infer(self, image: Image.Image, tokens: List[str], bboxes: List[List[int]], *, max_length: int = 512) -> Tuple[List[str], List[float]]:
        width, height = image.size
        norm_boxes = [normalize_box(b, width, height) for b in bboxes]

        encoding = self.processor(
            images=image,
            text=tokens,
            boxes=norm_boxes,
            truncation=True,
            padding="max_length",
            max_length=max_length,
            return_tensors="pt",
        )
        with torch.no_grad():
            logits = self.model(**encoding).logits.squeeze(0)

        probs = torch.softmax(logits, dim=-1)
        pred_ids = logits.argmax(-1).tolist()

        word_ids = None
        if hasattr(encoding, "word_ids"):
            try:
                word_ids = encoding.word_ids()
            except TypeError:
                word_ids = encoding.word_ids(batch_index=0)

        if word_ids:
            labels = _align_predictions(pred_ids, word_ids, self.id2label)
            score_map: Dict[int, float] = {}
            for idx, word_id in enumerate(word_ids):
                if word_id is None:
                    continue
                score_map.setdefault(word_id, float(probs[idx].max().item()))
            scores = [score_map.get(i, 0.0) for i in range(len(labels))]
            return labels, scores

        labels = [self.id2label.get(idx, "O") for idx in pred_ids[: len(tokens)]]
        scores = probs.max(dim=-1).values.tolist()[: len(tokens)]
        return labels, scores


def infer_tokens(
    image: Image.Image,
    tokens: List[str],
    bboxes: List[List[int]],
    *,
    model_dir: str,
    max_length: int = 512,
) -> Tuple[List[str], Optional[List[float]]]:
    inferer = TokenInferer.from_model_dir(model_dir)
    labels, scores = inferer.infer(image, tokens, bboxes, max_length=max_length)
    return labels, scores


@lru_cache(maxsize=4)
def _load_token_inferer(model_dir: str) -> TokenInferer:
    processor = AutoProcessor.from_pretrained(model_dir, apply_ocr=False)
    model = LayoutLMv3ForTokenClassification.from_pretrained(model_dir)
    model.eval()
    id2label = _load_labels(model_dir, model)
    return TokenInferer(model_dir=model_dir, processor=processor, model=model, id2label=id2label)


def warm_token_model(model_dir: str) -> None:
    _load_token_inferer(model_dir)
