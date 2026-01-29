"""Auto-label diploma fields in LayoutLM JSONL."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Dict, List


DATE_RE = re.compile(r"^(?:0?[1-9]|[12]\\d|3[01])[./-](?:0?[1-9]|1[0-2])[./-](?:19|20)\\d{2}$")
DEGREE_TYPES = {
    "diplom",
    "diploma",
    "bachelor",
    "master",
    "magister",
    "staatsexamen",
    "doctor",
    "doktor",
    "phd",
}
INSTITUTION_TOKENS = {
    "hochschule",
    "universitaet",
    "universität",
    "university",
    "college",
    "fachhochschule",
    "schule",
    "institute",
}
PROGRAM_TOKENS = {
    "studiengang",
    "fachrichtung",
    "program",
    "programme",
    "field",
    "course",
}
LOCATION_TOKENS = {"ort", "location", "issued"}
NAME_TOKENS = {"name", "holder", "inhaber", "inhaberin"}
STATUS_TOKENS = {"bestanden", "abgeschlossen", "graduated", "awarded", "conferred"}
CERTIFIED_TOKENS = {"beglaubigte", "beglaubigung", "certified", "copy"}
NUMBER_TOKENS = {"urkunden-nr", "urkunden", "diploma", "certificate", "no", "nr"}


def _norm(token: str) -> str:
    t = token.strip().lower()
    t = t.replace(":", "").replace(".", "").replace(",", "")
    return t


def _label_span(labels: List[str], start: int, end: int, prefix: str) -> int:
    if start >= end:
        return 0
    labels[start] = f"B-{prefix}"
    for i in range(start + 1, end):
        labels[i] = f"I-{prefix}"
    return end - start


def _label_record(record: Dict[str, object]) -> int:
    tokens = record.get("tokens") or []
    if not tokens:
        record["labels"] = []
        return 0

    labels = ["O"] * len(tokens)
    norm = [_norm(t) for t in tokens]
    labeled = 0

    for i, tok in enumerate(norm):
        if DATE_RE.match(tokens[i].strip()):
            labeled += _label_span(labels, i, i + 1, "DATE")
        if tok in DEGREE_TYPES:
            labeled += _label_span(labels, i, i + 1, "DEGREE_TYPE")
        if tok in STATUS_TOKENS:
            labeled += _label_span(labels, i, i + 1, "GRADUATION_STATUS")
        if tok in CERTIFIED_TOKENS:
            labeled += _label_span(labels, i, i + 1, "CERTIFIED_COPY_HINT")

    i = 0
    while i < len(norm):
        tok = norm[i]
        if tok in NAME_TOKENS:
            start = i + 1
            end = min(start + 3, len(tokens))
            labeled += _label_span(labels, start, end, "HOLDER_NAME")
            i = end
            continue
        if tok in INSTITUTION_TOKENS:
            start = i
            end = min(i + 4, len(tokens))
            labeled += _label_span(labels, start, end, "INSTITUTION_NAME")
            i = end
            continue
        if tok in PROGRAM_TOKENS:
            start = i + 1
            end = min(start + 4, len(tokens))
            labeled += _label_span(labels, start, end, "PROGRAM_OR_FIELD")
            i = end
            continue
        if tok in LOCATION_TOKENS:
            start = i + 1
            end = min(start + 2, len(tokens))
            labeled += _label_span(labels, start, end, "LOCATION")
            i = end
            continue
        if tok in NUMBER_TOKENS and i + 1 < len(tokens):
            labeled += _label_span(labels, i + 1, i + 2, "DIPLOMA_NUMBER")
            i += 2
            continue
        i += 1

    record["labels"] = labels
    return labeled


def main() -> None:
    parser = argparse.ArgumentParser(description="Auto-label diploma JSONL.")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    labeled = 0
    total = 0
    records = 0
    with args.input.open("r", encoding="utf-8") as f, args.output.open("w", encoding="utf-8") as out:
        for line in f:
            if not line.strip():
                continue
            record = json.loads(line)
            labeled += _label_record(record)
            total += len(record.get("tokens") or [])
            records += 1
            out.write(json.dumps(record, ensure_ascii=False) + "\n")

    pct = (labeled / total * 100) if total else 0.0
    print(f"Labeled {labeled}/{total} tokens ({pct:.2f}%) across {records} records.")


if __name__ == "__main__":
    main()
