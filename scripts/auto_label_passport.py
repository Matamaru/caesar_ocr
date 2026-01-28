"""Auto-label passport MRZ tokens in LayoutLM JSONL."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple


def _norm_mrz_token(text: str) -> str:
    t = text.strip().upper().replace(" ", "")
    return "".join(ch for ch in t if ch.isalnum() or ch == "<")


def _is_mrz_token(token: str) -> bool:
    if not token:
        return False
    return all(ch.isalnum() or ch == "<" for ch in token)


def _bbox_center_y(bbox: List[int]) -> float:
    return (bbox[1] + bbox[3]) / 2


def _build_lines(tokens: List[str], bboxes: List[List[int]], meta: List[Dict[str, object]] | None) -> List[Dict[str, object]]:
    if meta and len(meta) == len(tokens):
        lines: Dict[Tuple[int, int, int], List[Tuple[int, str, List[int], int]]] = {}
        for idx, tok in enumerate(tokens):
            m = meta[idx]
            key = (m.get("block_num", 0), m.get("par_num", 0), m.get("line_num", 0))
            lines.setdefault(key, []).append((idx, tok, bboxes[idx], m.get("word_num", 0)))

        out: List[Dict[str, object]] = []
        for items in lines.values():
            items.sort(key=lambda t: t[3])
            out.append(_assemble_line(items))
        return [l for l in out if l["line"]]

    # Fallback: cluster by y-position.
    items = []
    heights = []
    for idx, (tok, bbox) in enumerate(zip(tokens, bboxes)):
        y = _bbox_center_y(bbox)
        h = abs(bbox[3] - bbox[1])
        heights.append(h)
        items.append((idx, tok, bbox, y))
    if not items:
        return []
    heights.sort()
    median_h = heights[len(heights) // 2] if heights else 10
    tol = max(2.0, median_h * 0.6)

    items.sort(key=lambda t: t[3])
    lines: List[List[Tuple[int, str, List[int], float]]] = []
    current: List[Tuple[int, str, List[int], float]] = []
    current_y = None
    for item in items:
        if current_y is None or abs(item[3] - current_y) <= tol:
            current.append(item)
            current_y = item[3] if current_y is None else (current_y + item[3]) / 2
        else:
            lines.append(current)
            current = [item]
            current_y = item[3]
    if current:
        lines.append(current)

    out_lines: List[Dict[str, object]] = []
    for line_items in lines:
        line_items.sort(key=lambda t: t[2][0])
        out_lines.append(_assemble_line([(idx, tok, bbox, 0) for idx, tok, bbox, _y in line_items]))
    return [l for l in out_lines if l["line"]]


def _assemble_line(items: List[Tuple[int, str, List[int], int | float]]) -> Dict[str, object]:
    mapping: List[Tuple[int, int, int]] = []
    pos = 0
    parts: List[str] = []
    ys: List[float] = []
    indices: List[int] = []
    for idx, raw, bbox, _order in items:
        tok = _norm_mrz_token(raw)
        if not tok:
            continue
        start = pos
        end = pos + len(tok)
        mapping.append((idx, start, end))
        parts.append(tok)
        pos = end
        ys.append(_bbox_center_y(bbox))
        indices.append(idx)
    line = "".join(parts)
    return {
        "indices": indices,
        "line": line,
        "mapping": mapping,
        "y": sum(ys) / len(ys) if ys else 0.0,
    }


def _is_mrz_line(line: str) -> bool:
    if len(line) < 30:
        return False
    if not all(ch.isalnum() or ch == "<" for ch in line):
        return False
    if line.count("<") >= 2:
        return True
    digit_count = sum(ch.isdigit() for ch in line)
    return digit_count >= 10


def _select_mrz_lines(lines: List[Dict[str, object]]) -> List[Dict[str, object]]:
    if not lines:
        return []
    candidates = [l for l in lines if _is_mrz_line(l["line"])]
    if not candidates:
        return []
    candidates.sort(key=lambda l: (abs(len(l["line"]) - 44), l["y"]))
    chosen = candidates[:2]
    chosen.sort(key=lambda l: l["y"])
    return chosen


def _label_span(labels: List[str], mapping: List[Tuple[int, int, int]], start: int, end: int, prefix: str) -> int:
    if start >= end:
        return 0
    touched = 0
    first = True
    for idx, t_start, t_end in mapping:
        if t_end <= start or t_start >= end:
            continue
        if labels[idx] != "O":
            continue
        labels[idx] = f"B-{prefix}" if first else f"I-{prefix}"
        first = False
        touched += 1
    return touched


def _name_spans(line1: str) -> List[Tuple[str, int, int]]:
    spans: List[Tuple[str, int, int]] = []
    name_area = line1[5:]
    name_area = name_area.rstrip("<")
    if not name_area:
        return spans
    if "<<" in name_area:
        surname, given = name_area.split("<<", 1)
        if surname:
            spans.append(("SURNAME", 5, 5 + len(surname)))
        if given:
            given_start = 5 + len(surname) + 2
            spans.append(("GIVEN_NAMES", given_start, given_start + len(given)))
    else:
        spans.append(("SURNAME", 5, 5 + len(name_area)))
    return spans


def _label_record(record: Dict[str, object], *, mode: str) -> int:
    tokens = record.get("tokens") or []
    labels = ["O"] * len(tokens)

    bboxes = record.get("bboxes") or []
    if len(tokens) != len(bboxes) or not tokens:
        record["labels"] = labels
        return 0
    meta = record.get("token_meta")
    lines = _select_mrz_lines(_build_lines(tokens, bboxes, meta))
    if len(lines) < 2:
        record["labels"] = labels
        return 0

    line1 = lines[0]["line"]
    mapping1 = lines[0]["mapping"]
    line2 = lines[1]["line"]
    mapping2 = lines[1]["mapping"]

    labeled = 0
    if mode == "lines":
        labeled += _label_span(labels, mapping1, 0, len(line1), "MRZ_LINE1")
        labeled += _label_span(labels, mapping2, 0, len(line2), "MRZ_LINE2")
        record["labels"] = labels
        return labeled

    if len(line1) >= 44:
        labeled += _label_span(labels, mapping1, 0, 2, "DOC_CODE")
        labeled += _label_span(labels, mapping1, 2, 5, "ISSUING_COUNTRY")
        for prefix, start, end in _name_spans(line1):
            labeled += _label_span(labels, mapping1, start, end, prefix)

    if len(line2) >= 44:
        labeled += _label_span(labels, mapping2, 0, 9, "PASSPORT_NUMBER")
        labeled += _label_span(labels, mapping2, 9, 10, "PASSPORT_NUMBER_CHECK")
        labeled += _label_span(labels, mapping2, 10, 13, "NATIONALITY")
        labeled += _label_span(labels, mapping2, 13, 19, "BIRTH_DATE")
        labeled += _label_span(labels, mapping2, 19, 20, "BIRTH_DATE_CHECK")
        labeled += _label_span(labels, mapping2, 20, 21, "SEX")
        labeled += _label_span(labels, mapping2, 21, 27, "EXPIRY_DATE")
        labeled += _label_span(labels, mapping2, 27, 28, "EXPIRY_DATE_CHECK")
        labeled += _label_span(labels, mapping2, 28, 42, "PERSONAL_NUMBER")
        labeled += _label_span(labels, mapping2, 42, 43, "PERSONAL_NUMBER_CHECK")
        labeled += _label_span(labels, mapping2, 43, 44, "FINAL_CHECK")

    record["labels"] = labels
    return labeled


def main() -> None:
    parser = argparse.ArgumentParser(description="Auto-label passport MRZ fields in LayoutLM JSONL")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--mode",
        choices=["fields", "lines"],
        default="lines",
        help="Label MRZ line spans only (lines) or detailed fields (fields).",
    )
    args = parser.parse_args()

    labeled = 0
    total = 0
    records = 0

    with args.input.open("r", encoding="utf-8") as f, args.output.open("w", encoding="utf-8") as out:
        for line in f:
            if not line.strip():
                continue
            record = json.loads(line)
            labeled += _label_record(record, mode=args.mode)
            total += len(record.get("tokens") or [])
            records += 1
            out.write(json.dumps(record, ensure_ascii=False) + "\n")

    pct = (labeled / total * 100) if total else 0.0
    print(f"Labeled {labeled}/{total} tokens ({pct:.2f}%) across {records} records.")


if __name__ == "__main__":
    main()
