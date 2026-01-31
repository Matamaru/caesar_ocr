"""Evaluate extracted fields against a JSONL manifest."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from caesar_ocr.pipeline.analyze import analyze_document_bytes
from caesar_ocr.ocr.engine import detect_mrz_lines_from_text


@dataclass
class FieldStats:
    tp: int = 0
    fp: int = 0
    fn: int = 0

    def precision(self) -> float:
        return self.tp / (self.tp + self.fp) if (self.tp + self.fp) else 0.0

    def recall(self) -> float:
        return self.tp / (self.tp + self.fn) if (self.tp + self.fn) else 0.0

    def f1(self) -> float:
        p = self.precision()
        r = self.recall()
        return (2 * p * r / (p + r)) if (p + r) else 0.0


_CASE_INSENSITIVE_FIELDS = {
    "given_names",
    "surname",
    "holder_name_guess",
    "institution_guess",
    "program_guess",
    "degree_type_guess",
}
_IGNORE_UNEXPECTED_FIELDS = {"document_code", "dates_detected"}


def _normalize_value(value: Any, *, field: Optional[str] = None) -> List[str]:
    if value is None:
        return []
    if isinstance(value, list):
        values = [str(v).strip() for v in value if str(v).strip()]
    else:
        values = [str(value).strip()] if str(value).strip() else []
    values = [" ".join(v.split()) for v in values]
    if field in _CASE_INSENSITIVE_FIELDS:
        return [v.lower() for v in values]
    return values


def _values_match(expected: Any, predicted: Any, *, field: Optional[str] = None) -> bool:
    expected_vals = _normalize_value(expected, field=field)
    predicted_vals = _normalize_value(predicted, field=field)
    if not expected_vals:
        return False
    if not predicted_vals:
        return False
    return bool(set(expected_vals) & set(predicted_vals))


def _iter_manifest(path: Path) -> Iterable[Dict[str, Any]]:
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        yield json.loads(line)


def _eval_fields(expected: Dict[str, Any], predicted: Dict[str, Any], stats: Dict[str, FieldStats]) -> None:
    expected_fields = set(expected.keys())
    predicted_fields = {k for k in predicted.keys() if k not in _IGNORE_UNEXPECTED_FIELDS}

    for field in expected_fields:
        stat = stats.setdefault(field, FieldStats())
        if _values_match(expected.get(field), predicted.get(field), field=field):
            stat.tp += 1
        else:
            stat.fn += 1

    for field in predicted_fields - expected_fields:
        stat = stats.setdefault(field, FieldStats())
        stat.fp += 1


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate OCR field extraction against JSONL ground truth.")
    parser.add_argument("manifest", type=Path, help="JSONL with {path, expected} entries.")
    parser.add_argument("--layoutlm-model-dir", default=None, help="LayoutLM model directory")
    parser.add_argument("--layoutlm-token-model-dir", default=None, help="LayoutLM token classifier model directory")
    parser.add_argument("--lang", default="eng+deu", help="OCR language(s)")
    parser.add_argument("--layoutlm-lang", default=None, help="LayoutLM language flag")
    parser.add_argument("--regex-rules", default=None, help="Path to YAML regex rules")
    parser.add_argument("--regex-debug", action="store_true", help="Include regex debug trace in output")
    parser.add_argument("--output", type=Path, default=None, help="Write per-doc results JSONL")
    parser.add_argument("--include-ocr-text", action="store_true", help="Include OCR text in per-doc output")
    parser.add_argument("--ocr-text-max-len", type=int, default=2000, help="Max OCR text length to store")
    parser.add_argument("--include-mrz-lines", action="store_true", help="Include MRZ lines detected from OCR text")
    args = parser.parse_args()

    stats: Dict[str, FieldStats] = {}
    per_doc_rows: List[Dict[str, Any]] = []

    for item in _iter_manifest(args.manifest):
        path = Path(item["path"])
        expected = item.get("expected", {})
        payload = path.read_bytes()
        result = analyze_document_bytes(
            payload,
            layoutlm_model_dir=args.layoutlm_model_dir,
            lang=args.lang,
            layoutlm_lang=args.layoutlm_lang,
            regex_rules_path=args.regex_rules,
            regex_debug=args.regex_debug,
            layoutlm_token_model_dir=args.layoutlm_token_model_dir,
        )
        predicted_fields = result.ocr.fields
        _eval_fields(expected, predicted_fields, stats)

        row = {
            "path": str(path),
            "expected": expected,
            "predicted": predicted_fields,
            "doc_type": result.ocr.doc_type,
        }
        if args.include_ocr_text:
            text = result.ocr.ocr_text or ""
            if args.ocr_text_max_len and len(text) > args.ocr_text_max_len:
                text = text[: args.ocr_text_max_len]
            row["ocr_text"] = text
        if args.include_mrz_lines:
            row["mrz_lines_detected"] = detect_mrz_lines_from_text(result.ocr.ocr_text or "")
        per_doc_rows.append(row)

    summary = {}
    for field, stat in sorted(stats.items()):
        summary[field] = {
            "tp": stat.tp,
            "fp": stat.fp,
            "fn": stat.fn,
            "precision": stat.precision(),
            "recall": stat.recall(),
            "f1": stat.f1(),
        }

    print(json.dumps({"summary": summary}, indent=2))

    if args.output:
        args.output.write_text("\n".join(json.dumps(row) for row in per_doc_rows) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
