"""Canonical output schemas for pipeline outputs and datasets."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Dict, List, Optional


@dataclass
class OcrToken:
    text: str
    bbox: List[int]
    start: Optional[int] = None
    end: Optional[int] = None
    conf: Optional[float] = None
    label: Optional[str] = None
    label_score: Optional[float] = None


@dataclass
class OcrPage:
    page: int
    width: int
    height: int
    text: str
    tokens: List[OcrToken] = field(default_factory=list)


@dataclass
class OcrDocument:
    doc_id: Optional[str]
    doc_type: str
    language: Optional[str]
    pages: List[OcrPage] = field(default_factory=list)
    fields: Dict[str, object] = field(default_factory=dict)


@dataclass
class LayoutLMClassification:
    label: str
    scores: Dict[str, float]


@dataclass
class LayoutLMTokenClassification:
    labels: List[str]
    scores: Optional[List[float]] = None


@dataclass
class PipelineResult:
    ocr: OcrDocument
    layoutlm: Optional[LayoutLMClassification] = None
    layoutlm_tokens: Optional[LayoutLMTokenClassification] = None

    def to_dict(self) -> dict:
        return asdict(self)
