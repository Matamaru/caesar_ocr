"""Bootstrap token labels for Fehlerprotokoll JSONL using simple regex parsing."""

from __future__ import annotations

import argparse
import json
import re
import sqlite3
from pathlib import Path
from typing import List, Tuple


DATE_TOKEN_RE = re.compile(r"^\d{2}\.\d{2}\.\d{4}$")
TIME_TOKEN_RE = re.compile(r"^\d{2}:\d{2}$")
IK_RE = re.compile(r"^ik:?$", re.IGNORECASE)


def _norm_token(text: str) -> str:
    t = text.strip().lower()
    t = t.replace("–", "-")
    t = re.sub(r"^[^a-z0-9§]+|[^a-z0-9§]+$", "", t)
    t = t.replace("sgbxi", "sgb xi")
    return t


def _find_seq(tokens: List[str], target: List[str], start: int) -> Tuple[int, int] | None:
    if not target:
        return None
    tlen = len(target)
    for i in range(start, len(tokens) - tlen + 1):
        if tokens[i : i + tlen] == target:
            return i, i + tlen
    return None


def _label_span(labels: List[str], start: int, end: int, prefix: str) -> None:
    if start < 0 or end <= start:
        return
    labels[start] = f"B-{prefix}"
    for i in range(start + 1, end):
        labels[i] = f"I-{prefix}"


def _split_value(value: str) -> List[str]:
    norm = _norm_token(value)
    if not norm:
        return []
    # Split on whitespace and slashes/dashes to handle OCR quirks.
    parts = re.split(r"[\s/,-]+", norm)
    return [p for p in parts if p]


def _find_next(tokens: List[str], start: int, predicate) -> int | None:
    for i in range(start, len(tokens)):
        if predicate(tokens[i]):
            return i
    return None


def _is_date_token(token: str) -> bool:
    return bool(DATE_TOKEN_RE.match(token))


def _is_time_token(token: str) -> bool:
    return bool(TIME_TOKEN_RE.match(token))


def _is_zip(token: str) -> bool:
    return token.isdigit() and len(token) == 5


def _label_record(record: dict, *, company_name: str | None) -> int:
    tokens = record.get("tokens") or []
    if not tokens:
        record["labels"] = []
        return 0

    norm_tokens = [_norm_token(t) for t in tokens]
    labels = ["O"] * len(tokens)

    cursor = 0
    labeled = 0

    # Company name
    if company_name:
        company_tokens = _split_value(company_name)
        found = _find_seq(norm_tokens, company_tokens, 0)
        if found:
            start, end = found
            _label_span(labels, start, end, "COMPANY_NAME")
            labeled += (end - start)

    # Report date (token after "datum")
    for i, tok in enumerate(norm_tokens):
            if tok in ("datum", "date"):
                j = _find_next(norm_tokens, i + 1, _is_date_token)
                if j is not None:
                    _label_span(labels, j, j + 1, "REPORT_DATE")
                    labeled += 1
                # Optional time token right after date
                k = _find_next(norm_tokens, j + 1, _is_time_token)
                if k is not None and k == j + 1:
                    _label_span(labels, k, k + 1, "REPORT_TIME")
                    labeled += 1
            break

    # IK number: look for token "ik" then next numeric token
    for i, tok in enumerate(norm_tokens):
        if IK_RE.match(tok):
            j = _find_next(norm_tokens, i + 1, lambda t: t.isdigit())
            if j is not None:
                _label_span(labels, j, j + 1, "IK")
                labeled += 1
            break

    # Address: zipcode + town (simple heuristic)
    for i, tok in enumerate(norm_tokens):
        if _is_zip(tok):
            _label_span(labels, i, i + 1, "ZIPCODE")
            labeled += 1
            town_idx = _find_next(norm_tokens, i + 1, lambda t: t.isalpha())
            if town_idx is not None and town_idx == i + 1:
                _label_span(labels, town_idx, town_idx + 1, "TOWN")
                labeled += 1
            break

    i = 0
    while i < len(norm_tokens):
        tok = norm_tokens[i]
        if tok.startswith("rechnung"):
            # Position number = previous numeric token
            pos_idx = i - 1
            while pos_idx >= 0 and not norm_tokens[pos_idx].isdigit():
                pos_idx -= 1
            if pos_idx >= 0:
                _label_span(labels, pos_idx, pos_idx + 1, "POSITION_NR")
                labeled += 1

            # Find "für"
            fuer_idx = _find_next(norm_tokens, i + 1, lambda t: t in ("für", "fur"))
            if fuer_idx is not None:
                # Last name and first name
                last_idx = _find_next(norm_tokens, fuer_idx + 1, lambda t: t)
                first_idx = _find_next(
                    norm_tokens,
                    (last_idx + 1) if last_idx is not None else fuer_idx + 1,
                    lambda t: t,
                )
                if last_idx is not None:
                    _label_span(labels, last_idx, last_idx + 1, "CUSTOMER_LAST")
                    labeled += 1
                if first_idx is not None:
                    _label_span(labels, first_idx, first_idx + 1, "CUSTOMER_FIRST")
                    labeled += 1

                # Date range (two date tokens)
                start_date_idx = _find_next(norm_tokens, fuer_idx + 1, _is_date_token)
                end_date_idx = None
                if start_date_idx is not None:
                    end_date_idx = _find_next(norm_tokens, start_date_idx + 1, _is_date_token)
                if start_date_idx is not None:
                    _label_span(labels, start_date_idx, start_date_idx + 1, "PERIOD_START")
                    labeled += 1
                if end_date_idx is not None:
                    _label_span(labels, end_date_idx, end_date_idx + 1, "PERIOD_END")
                    labeled += 1

                # Service span (from § to XI)
                service_start = _find_next(
                    norm_tokens, fuer_idx + 1, lambda t: "§" in t or t.startswith("§") or t.startswith("sgb")
                )
                if service_start is not None:
                    service_end = _find_next(
                        norm_tokens,
                        service_start + 1,
                        lambda t: t in ("xi", "sgb", "sgbxi"),
                    )
                    if service_end is not None:
                        _label_span(labels, service_start, service_end + 1, "SERVICE")
                        labeled += (service_end + 1 - service_start)

                        # Error type starts at next "für" after service and ends at "drucken"
                        err_start = _find_next(norm_tokens, service_end + 1, lambda t: t in ("für", "fur"))
                        if err_start is not None:
                            err_end = _find_next(norm_tokens, err_start + 1, lambda t: t.startswith("druck"))
                            if err_end is not None:
                                _label_span(labels, err_start, err_end + 1, "ERROR_TYPE")
                                labeled += (err_end + 1 - err_start)
        i += 1

    record["labels"] = labels
    return labeled


def _load_company_name(db_path: Path) -> str | None:
    if not db_path.exists():
        return None
    conn = sqlite3.connect(db_path)
    try:
        row = conn.execute("SELECT name FROM company ORDER BY id LIMIT 1").fetchone()
    finally:
        conn.close()
    if not row:
        return None
    return row[0]


def main() -> None:
    parser = argparse.ArgumentParser(description="Auto-label Fehlerprotokoll LayoutLM JSONL.")
    parser.add_argument("--input", type=Path, required=True, help="Input JSONL")
    parser.add_argument("--output", type=Path, required=True, help="Output JSONL")
    parser.add_argument("--company-name", type=str, default=None, help="Override company name for labeling")
    parser.add_argument(
        "--company-db",
        type=Path,
        default=Path("apps/company_universe/db/company.sqlite"),
        help="SQLite DB to read company name from (if not provided)",
    )
    args = parser.parse_args()

    company_name = args.company_name or _load_company_name(args.company_db)

    labeled_tokens = 0
    total_tokens = 0
    records = 0

    with args.input.open("r", encoding="utf-8") as f, args.output.open("w", encoding="utf-8") as out:
        for line in f:
            if not line.strip():
                continue
            record = json.loads(line)
            labeled_tokens += _label_record(record, company_name=company_name)
            total_tokens += len(record.get("tokens") or [])
            out.write(json.dumps(record, ensure_ascii=False) + "\n")
            records += 1

    coverage = (labeled_tokens / total_tokens) if total_tokens else 0.0
    print(f"Labeled {labeled_tokens}/{total_tokens} tokens ({coverage:.2%}) across {records} records.")


if __name__ == "__main__":
    main()
