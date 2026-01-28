"""Combined OCR + LayoutLM tool entry point."""

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import List, Optional

from ..layoutlm.infer import LayoutLMResult, analyze_bytes_layoutlm
from ..layoutlm.token_infer import TokenInferer
from ..ocr.engine import OcrResult, analyze_pages
from ..io.loaders import load_images_from_bytes
from ..regex.engine import load_rules, run_rules
from ..regex.classify import infer_present_docs
from .schemas import LayoutLMClassification, LayoutLMTokenClassification, OcrDocument, OcrPage, OcrToken, PipelineResult


def _postprocess_position_labels(
    page_tokens: List[dict],
    labels: List[str],
    scores: List[float],
) -> tuple[List[str], List[float]]:
    if not labels:
        return labels, scores
    out_labels = list(labels)
    out_scores = list(scores) if scores else []

    seen_lines = set()
    last_pos = 0
    for i, tok in enumerate(page_tokens):
        if i >= len(out_labels):
            break
        if out_labels[i] != "B-POSITION_NR":
            continue
        text = str(tok.get("text", "")).strip()
        if not text.isdigit():
            out_labels[i] = "O"
            if i < len(out_scores):
                out_scores[i] = 0.0
            continue

        line_key = (
            tok.get("block_num"),
            tok.get("par_num"),
            tok.get("line_num"),
        )
        if any(v is not None for v in line_key):
            if line_key in seen_lines:
                out_labels[i] = "O"
                if i < len(out_scores):
                    out_scores[i] = 0.0
                continue
            seen_lines.add(line_key)

        pos_val = int(text)
        if pos_val <= last_pos:
            out_labels[i] = "O"
            if i < len(out_scores):
                out_scores[i] = 0.0
            continue
        last_pos = pos_val

    return out_labels, out_scores


@dataclass
class AssistantToolResult:
    """Bundle OCR and optional LayoutLM results in a serializable form."""
    ocr: OcrResult
    layoutlm: Optional[LayoutLMResult] = None
    schema: Optional[PipelineResult] = None

    def to_dict(self, *, schema: bool = True) -> dict:
        if schema and self.schema is not None:
            return self.schema.to_dict()
        data = {"ocr": asdict(self.ocr)}
        if self.layoutlm is not None:
            data["layoutlm"] = asdict(self.layoutlm)
        return data


def analyze_document_bytes(
    file_bytes: bytes,
    layoutlm_model_dir: Optional[str] = None,
    *,
    lang: str = "eng+deu",
    layoutlm_lang: Optional[str] = None,
    regex_rules_path: Optional[str] = None,
    regex_debug: bool = False,
    layoutlm_token_model_dir: Optional[str] = None,
) -> AssistantToolResult:
    """Analyze bytes with OCR and optional LayoutLM classifier."""
    pages = load_images_from_bytes(file_bytes, dpi=300)
    ocr_result = analyze_pages(pages, lang=lang)
    page_items = []
    layoutlm_result = None
    if layoutlm_model_dir:
        layoutlm_result = analyze_bytes_layoutlm(
            file_bytes,
            model_dir=layoutlm_model_dir,
            lang=layoutlm_lang,
        )
    if regex_rules_path:
        rules = load_rules(Path(regex_rules_path))
        regex_fields = run_rules(ocr_result.ocr_text, rules, debug=regex_debug)
        if regex_fields:
            ocr_result.fields.update(regex_fields)
    # Always add lightweight doc type hints from OCR text.
    doc_hints = infer_present_docs(ocr_result.ocr_text)
    if doc_hints:
        ocr_result.fields.setdefault("doc_hints", sorted(doc_hints))
    layout = None
    layoutlm_tokens = None
    token_labels_by_page: dict[int, List[str]] = {}
    token_scores_by_page: dict[int, List[float]] = {}
    if layoutlm_token_model_dir and pages:
        all_labels: List[str] = []
        all_scores: List[float] = []
        inferer = TokenInferer.from_model_dir(layoutlm_token_model_dir)
        for page in pages:
            page_tokens = [t for t in ocr_result.tokens if t.get("page", page.page) == page.page]
            token_texts = [t.get("text", "") for t in page_tokens]
            token_boxes = [t.get("bbox") or [0, 0, 0, 0] for t in page_tokens]
            labels, scores = inferer.infer(page.image, token_texts, token_boxes)
            token_labels_by_page[page.page] = labels
            if scores:
                token_scores_by_page[page.page] = scores
                all_scores.extend(scores)
            all_labels.extend(labels)
        layoutlm_tokens = LayoutLMTokenClassification(labels=all_labels, scores=all_scores or None)

    if pages:
        for idx, page in enumerate(pages):
            page_tokens = [t for t in ocr_result.tokens if t.get("page", page.page) == page.page]
            labels = token_labels_by_page.get(page.page, [])
            scores = token_scores_by_page.get(page.page, [])
            labels, scores = _postprocess_position_labels(page_tokens, labels, scores)
            tokens = [
                OcrToken(
                    text=t.get("text", ""),
                    bbox=t.get("bbox") or [0, 0, 0, 0],
                    start=t.get("start"),
                    end=t.get("end"),
                    conf=t.get("conf"),
                    label=labels[i] if i < len(labels) else None,
                    label_score=scores[i] if i < len(scores) else None,
                )
                for i, t in enumerate(page_tokens)
            ]
            text = ocr_result.page_texts[idx] if idx < len(ocr_result.page_texts) else ocr_result.ocr_text
            page_items.append(
                OcrPage(
                    page=page.page,
                    width=page.width,
                    height=page.height,
                    text=text,
                    tokens=tokens,
                )
            )
    else:
        page_items.append(OcrPage(page=1, width=0, height=0, text=ocr_result.ocr_text))
    if layoutlm_result is not None:
        layout = LayoutLMClassification(label=layoutlm_result.doc_type, scores=layoutlm_result.scores)
    schema = PipelineResult(
        ocr=OcrDocument(
            doc_id=None,
            doc_type=ocr_result.doc_type,
            language=lang,
            pages=page_items,
            fields=ocr_result.fields,
        ),
        layoutlm=layout,
        layoutlm_tokens=layoutlm_tokens,
    )
    return AssistantToolResult(ocr=ocr_result, layoutlm=layoutlm_result, schema=schema)
