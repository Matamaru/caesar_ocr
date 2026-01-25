"""CLI for LayoutLMv3 token classification inference."""

from __future__ import annotations

import argparse
import json
import pathlib
from typing import Dict, List, Tuple

from PIL import Image
import torch
import pytesseract
from transformers import AutoProcessor, LayoutLMv3ForTokenClassification


def _read_jsonl(path: pathlib.Path) -> List[Dict[str, object]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def _normalize_box(box: List[int], width: int, height: int) -> List[int]:
    x0, y0, x1, y1 = box
    return [
        max(0, min(1000, int(1000 * x0 / width))),
        max(0, min(1000, int(1000 * y0 / height))),
        max(0, min(1000, int(1000 * x1 / width))),
        max(0, min(1000, int(1000 * y1 / height))),
    ]


def _load_labels(model_dir: pathlib.Path, model) -> Dict[int, str]:
    labels_path = model_dir / "labels.json"
    if labels_path.exists():
        labels = json.loads(labels_path.read_text())
        return {idx: label for idx, label in enumerate(labels)}
    return model.config.id2label


def _ocr_tokens(image: Image.Image, *, lang: str) -> Tuple[str, List[str], List[List[int]]]:
    data = pytesseract.image_to_data(image, lang=lang, output_type=pytesseract.Output.DICT)
    tokens: List[str] = []
    bboxes: List[List[int]] = []
    n = len(data["text"])
    for i in range(n):
        text = (data["text"][i] or "").strip()
        if not text:
            continue
        left = int(data["left"][i])
        top = int(data["top"][i])
        width = int(data["width"][i])
        height = int(data["height"][i])
        tokens.append(text)
        bboxes.append([left, top, left + width, top + height])
    full_text = " ".join(tokens)
    return full_text, tokens, bboxes


def _records_from_file(path: pathlib.Path, page: int | None, *, lang: str) -> List[Dict[str, object]]:
    if path.suffix.lower() == ".jsonl":
        return _read_jsonl(path)

    if path.suffix.lower() == ".pdf":
        try:
            from pdf2image import convert_from_path
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise SystemExit("pdf2image is required for PDF input. Install it and try again.") from exc

        pages = convert_from_path(path)
        if not pages:
            raise SystemExit("No pages rendered from PDF.")
        indices = [page - 1] if page else list(range(len(pages)))
        records: List[Dict[str, object]] = []
        for idx in indices:
            if idx < 0 or idx >= len(pages):
                raise SystemExit(f"Page {idx + 1} is out of range (1-{len(pages)}).")
            image = pages[idx].convert("RGB")
            full_text, tokens, bboxes = _ocr_tokens(image, lang=lang)
            records.append(
                {
                    "id": None,
                    "image": str(path),
                    "text": full_text,
                    "doc_id": path.stem,
                    "page": idx + 1,
                    "tokens": tokens,
                    "bboxes": bboxes,
                    "labels": [],
                    "spans": [],
                }
            )
        return records

    image = Image.open(path).convert("RGB")
    full_text, tokens, bboxes = _ocr_tokens(image, lang=lang)
    return [
        {
            "id": None,
            "image": str(path),
            "text": full_text,
            "doc_id": path.stem,
            "page": 1,
            "tokens": tokens,
            "bboxes": bboxes,
            "labels": [],
            "spans": [],
        }
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description="Run LayoutLMv3 token classifier inference")
    parser.add_argument("--model-dir", type=pathlib.Path, required=True)
    parser.add_argument("--input", type=pathlib.Path, required=True, help="JSONL with tokens/bboxes, or an image/PDF")
    parser.add_argument("--page", type=int, default=None, help="PDF page number (1-based). Default: all pages")
    parser.add_argument("--output", type=pathlib.Path, required=True, help="Output JSONL with predictions")
    parser.add_argument("--lang", default="eng+deu", help="OCR language(s) for image/PDF input")
    args = parser.parse_args()

    records = _records_from_file(args.input, args.page, lang=args.lang)
    processor = AutoProcessor.from_pretrained(args.model_dir, apply_ocr=False)
    model = LayoutLMv3ForTokenClassification.from_pretrained(args.model_dir)
    model.eval()

    id2label = _load_labels(args.model_dir, model)

    outputs = []
    for rec in records:
        image_path = pathlib.Path(rec["image"])
        if image_path.suffix.lower() == ".pdf":
            try:
                from pdf2image import convert_from_path
            except ImportError as exc:  # pragma: no cover - optional dependency
                raise SystemExit("pdf2image is required for PDF input. Install it and try again.") from exc
            pages = convert_from_path(image_path)
            page_idx = int(rec.get("page") or 1) - 1
            if page_idx < 0 or page_idx >= len(pages):
                raise SystemExit(f"Page {page_idx + 1} is out of range (1-{len(pages)}).")
            image = pages[page_idx].convert("RGB")
        else:
            image = Image.open(image_path).convert("RGB")
        width, height = image.size
        tokens = rec["tokens"]
        boxes = [_normalize_box(b, width, height) for b in rec["bboxes"]]

        encoding = processor(
            images=image,
            text=tokens,
            boxes=boxes,
            truncation=True,
            padding="max_length",
            max_length=512,
            return_tensors="pt",
        )
        with torch.no_grad():
            logits = model(**encoding).logits.squeeze(0)

        pred_ids = logits.argmax(-1).tolist()
        labels = [id2label.get(idx, "O") for idx in pred_ids[: len(tokens)]]

        out = dict(rec)
        out["labels"] = labels
        outputs.append(out)

    args.output.write_text("\n".join(json.dumps(rec, ensure_ascii=True) for rec in outputs))
    print(f"Wrote {len(outputs)} records to {args.output}")


if __name__ == "__main__":
    main()
