"""JSONL dataset loader and validation."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Iterable, List, Optional


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
