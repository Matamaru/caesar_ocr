"""JSONL dataset loader and validation."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional


@dataclass
class LayoutLMTokenRecord:
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
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            data = json.loads(line)
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
    errors: List[str] = []
    if not rec.text:
        errors.append("text is empty")
    if len(rec.tokens) != len(rec.bboxes):
        errors.append("tokens and bboxes length mismatch")
    if rec.labels and len(rec.labels) != len(rec.tokens):
        errors.append("labels and tokens length mismatch")
    for box in rec.bboxes:
        if not isinstance(box, list) or len(box) != 4:
            errors.append("invalid bbox entry")
            break
    return errors


def quality_checks(rec: LayoutLMTokenRecord) -> Dict[str, object]:
    """Return data quality stats for a single record."""
    issues: List[str] = []

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

    return {
        "issues": issues,
        "label_coverage": coverage,
        "bbox_oob": bbox_oob,
        "num_tokens": len(rec.tokens),
        "num_labels": len(rec.labels),
    }
