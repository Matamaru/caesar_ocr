"""
LayoutLM helpers for document classification and scoring.

infer means to deduce or conclude information from evidence and reasoning rather than from explicit statements.
"""

from dataclasses import dataclass
from functools import lru_cache
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
    """Load the first page image from PDF/image bytes."""
    pages = load_images_from_bytes(file_bytes, dpi=300)
    return pages[0].image


def _build_label_maps(labels: Optional[List[str]]) -> Tuple[Optional[Dict[str, int]], Optional[Dict[int, str]]]:
    """
    Build label to ID and ID to label mappings.
    
    :param labels: List of label strings.
    :return: Tuple of (label2id, id2label) dictionaries or (None, None) if labels is None.
    """
    # If no labels provided, return None
    if not labels:
        return None, None
    # Create mappings between labels and IDs
    label2id = {lbl: i for i, lbl in enumerate(labels)}
    id2label = {i: lbl for lbl, i in label2id.items()}
    return label2id, id2label


def _resolve_device(device: Optional[str]) -> Optional[str]:
    if device is None and torch.cuda.is_available():
        return "cuda"
    return device


@lru_cache(maxsize=4)
def _load_layoutlm_components(
    model_dir: str,
    processor_name: str,
    labels_key: Optional[Tuple[str, ...]],
    device: Optional[str],
):
    label2id, id2label = _build_label_maps(list(labels_key) if labels_key else None)
    processor = AutoProcessor.from_pretrained(processor_name)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_dir,
        num_labels=len(labels_key) if labels_key else None,
        id2label=id2label,
        label2id=label2id,
    )
    model.eval()
    if device:
        model.to(device)
    return processor, model


def warm_layoutlm_model(
    model_dir: str,
    *,
    processor_name: Optional[str] = None,
    labels: Optional[List[str]] = None,
    device: Optional[str] = None,
) -> None:
    resolved_device = _resolve_device(device)
    labels_key = tuple(labels) if labels else None
    _load_layoutlm_components(model_dir, processor_name or model_dir, labels_key, resolved_device)


def analyze_bytes_layoutlm(
    file_bytes: bytes,
    model_dir: str,
    *,
    processor_name: Optional[str] = None,
    labels: Optional[List[str]] = None,
    lang: Optional[str] = None,
    device: Optional[str] = None,
) -> LayoutLMResult:
    """
    Run LayoutLMv3 sequence classification on a PDF/image payload.
    
    :param file_bytes: Bytes of the input PDF or image file.
    :param model_dir: Directory of the pretrained LayoutLM model.
    :param processor_name: Optional name of the processor to load, defaults to model_dir if not provided.
    :param labels: Optional list of label strings for classification.
    :param lang: Optional language code for the processor.
    :param device: Optional device string (e.g., "cpu" or "cuda").
    :return: LayoutLMResult containing classification results.
    """
    # Load the first page image from the input bytes
    image = _load_image_from_bytes(file_bytes)

    resolved_device = _resolve_device(device)
    labels_key = tuple(labels) if labels else None
    processor, model = _load_layoutlm_components(
        model_dir,
        processor_name or model_dir,
        labels_key,
        resolved_device,
    )

    # Prepare inputs and run inference
    encoding = processor(images=image, return_tensors="pt", lang=lang)
    if resolved_device:
        encoding = {k: v.to(resolved_device) for k, v in encoding.items()}

    # Disable gradient calculations for inference
    # Necessary for efficiency and to avoid unnecessary memory usage
    with torch.no_grad():
        logits = model(**encoding).logits.squeeze(0)

    # Post-process logits to obtain probabilities and predicted label
    probs = torch.softmax(logits, dim=-1).tolist()
    
    # Map model's ID to label string
    model_id2label = model.config.id2label
    
    # scores is a mapping from label string to probability
    scores = {model_id2label[i]: float(p) for i, p in enumerate(probs)}
    
    # Get the label ID with the highest probability
    label_id = int(torch.argmax(logits).item())

    # Get the corresponding document type label
    doc_type = model_id2label.get(label_id, str(label_id))

    # Return the classification result as instance of LayoutLMResult
    return LayoutLMResult(doc_type=doc_type, label_id=label_id, scores=scores)
