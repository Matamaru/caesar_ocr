"""Combined OCR + LayoutLM tool entry point."""

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional

from ..layoutlm.infer import LayoutLMResult, analyze_bytes_layoutlm
from ..ocr.engine import OcrResult, analyze_bytes
from ..io.loaders import load_images_from_bytes
from ..regex.engine import load_rules, run_rules
from .schemas import LayoutLMClassification, OcrDocument, OcrPage, OcrToken, PipelineResult


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
) -> AssistantToolResult:
    """Analyze bytes with OCR and optional LayoutLM classifier."""
    ocr_result = analyze_bytes(file_bytes, lang=lang)
    pages = load_images_from_bytes(file_bytes, dpi=300)
    page_items = []
    if pages:
        for idx, page in enumerate(pages):
            page_tokens = [t for t in ocr_result.tokens if t.get("page", page.page) == page.page]
            tokens = [
                OcrToken(
                    text=t.get("text", ""),
                    bbox=t.get("bbox") or [0, 0, 0, 0],
                    start=t.get("start"),
                    end=t.get("end"),
                    conf=t.get("conf"),
                )
                for t in page_tokens
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
    layout = None
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
    )
    return AssistantToolResult(ocr=ocr_result, layoutlm=layoutlm_result, schema=schema)
