"""JSONL dataset loader and validation."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional


@dataclass
class LayoutLMTokenRecord:
    """
    A single record in a LayoutLM token-level dataset.

    :param id: Optional unique identifier for the record.
    :param image: Optional path or identifier for the associated image.
    :param text: Full text of the document/page.
    :param doc_id: Optional document identifier.
    :param page: Optional page number.
    :param tokens: List of token strings.
    :param bboxes: List of bounding boxes corresponding to tokens.
    :param labels: List of labels corresponding to tokens.
    :param spans: List of span dictionaries with additional metadata.
    """
    id: Optional[str]
    image: Optional[str]
    text: str
    doc_id: Optional[str]
    page: Optional[int]
    tokens: List[str]
    bboxes: List[List[int]]
    labels: List[str]
    spans: List[dict]


def iter_jsonl(path) -> Iterable[LayoutLMTokenRecord]:
    """
    Iterate over a JSONL file and yield LayoutLMTokenRecord instances.
    
    :param path: Path to the JSONL file.
    :return: An iterable of LayoutLMTokenRecord objects.
    """
    # Open the JSONL file and read line by line
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            data = json.loads(line)
            # Yield a LayoutLMTokenRecord instance for each line
            yield LayoutLMTokenRecord(
                id=data.get("id"),
                image=data.get("image"),
                text=data.get("text", ""),
                doc_id=data.get("doc_id"),
                page=data.get("page"),
                tokens=data.get("tokens") or [],
                bboxes=data.get("bboxes") or [],
                labels=data.get("labels") or [],
                spans=data.get("spans") or [],
            )


def validate_record(rec: LayoutLMTokenRecord) -> List[str]:
    """
    Validate a single LayoutLMTokenRecord and return a list of error messages.
    
    :param rec: LayoutLMTokenRecord to validate.
    :return: List of error messages, empty if no errors.
    """
    errors: List[str] = []
    # Check for common validation issues

    # rec.text should not be empty. rec is expected to have tokens and bboxes of equal length.
    if not rec.text:
        errors.append("text is empty")
    
    # tokens and bboxes must be non-empty and of equal length
    if len(rec.tokens) != len(rec.bboxes):
        errors.append("tokens and bboxes length mismatch")

    # labels, if present, must match tokens length
    if rec.labels and len(rec.labels) != len(rec.tokens):
        errors.append("labels and tokens length mismatch")

    # Each bbox must be a list of four integers
    # the four integers represent [x0, y0, x1, y1]
    for box in rec.bboxes:
        if not isinstance(box, list) or len(box) != 4:
            errors.append("invalid bbox entry")
            break
    return errors


def quality_checks(rec: LayoutLMTokenRecord) -> Dict[str, object]:
    """
    Return data quality stats for a single record.
    
    :param rec: LayoutLMTokenRecord to check.
    :return: Dictionary with quality metrics.
    """
    issues: List[str] = []

    # Basic presence checks
    if not rec.tokens:
        issues.append("no_tokens")
    if not rec.bboxes:
        issues.append("no_bboxes")
    if not rec.labels:
        issues.append("no_labels")
    if len(rec.tokens) != len(rec.bboxes):
        issues.append("len_mismatch_tokens_bboxes")
    if rec.labels and len(rec.labels) != len(rec.tokens):
        issues.append("len_mismatch_tokens_labels")

    # bbox bounds in LayoutLM normalized space (0..1000)
    bbox_oob = 0
    bbox_space = "normalized"
    if rec.bboxes:
        max_coord = max(max(box) for box in rec.bboxes if len(box) == 4)
        if max_coord > 1000:
            bbox_space = "pixel"
        else:
            for box in rec.bboxes:
                if len(box) != 4:
                    continue
                x0, y0, x1, y1 = box
                if not (0 <= x0 <= 1000 and 0 <= y0 <= 1000 and 0 <= x1 <= 1000 and 0 <= y1 <= 1000):
                    bbox_oob += 1
            if bbox_oob:
                issues.append("bbox_out_of_bounds")

    # label coverage ratio
    label_count = sum(1 for lbl in rec.labels if lbl and lbl != "O")
    total = len(rec.labels)
    coverage = (label_count / total) if total else 0.0

    # Return the quality metrics
    return {
        "issues": issues,
        "label_coverage": coverage,
        "bbox_oob": bbox_oob,
        "bbox_space": bbox_space,
        "num_tokens": len(rec.tokens),
        "num_labels": len(rec.labels),
    }
