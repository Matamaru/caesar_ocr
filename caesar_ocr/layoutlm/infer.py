"""LayoutLM helpers for document classification and scoring."""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

from ..io.loaders import load_images_from_bytes
import torch
from transformers import AutoProcessor, AutoModelForSequenceClassification


@dataclass
class LayoutLMResult:
    """Result of LayoutLM document classification."""
    doc_type: str
    label_id: int
    scores: Dict[str, float]


def _load_image_from_bytes(file_bytes: bytes) -> object:
    pages = load_images_from_bytes(file_bytes, dpi=300)
    return pages[0].image


def _build_label_maps(labels: Optional[List[str]]) -> Tuple[Optional[Dict[str, int]], Optional[Dict[int, str]]]:
    if not labels:
        return None, None
    label2id = {lbl: i for i, lbl in enumerate(labels)}
    id2label = {i: lbl for lbl, i in label2id.items()}
    return label2id, id2label


def analyze_bytes_layoutlm(
    file_bytes: bytes,
    model_dir: str,
    *,
    processor_name: Optional[str] = None,
    labels: Optional[List[str]] = None,
    lang: Optional[str] = None,
    device: Optional[str] = None,
) -> LayoutLMResult:
    """Run LayoutLMv3 sequence classification on a PDF/image payload."""
    image = _load_image_from_bytes(file_bytes)

    label2id, id2label = _build_label_maps(labels)
    processor = AutoProcessor.from_pretrained(processor_name or model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_dir,
        num_labels=len(labels) if labels else None,
        id2label=id2label,
        label2id=label2id,
    )
    model.eval()

    if device is None and torch.cuda.is_available():
        device = "cuda"
    if device:
        model.to(device)

    encoding = processor(images=image, return_tensors="pt", lang=lang)
    if device:
        encoding = {k: v.to(device) for k, v in encoding.items()}

    with torch.no_grad():
        logits = model(**encoding).logits.squeeze(0)

    probs = torch.softmax(logits, dim=-1).tolist()
    model_id2label = model.config.id2label
    scores = {model_id2label[i]: float(p) for i, p in enumerate(probs)}
    label_id = int(torch.argmax(logits).item())
    doc_type = model_id2label.get(label_id, str(label_id))

    return LayoutLMResult(doc_type=doc_type, label_id=label_id, scores=scores)
