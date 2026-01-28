"""Bootstrap token labels for CV JSONL using simple token heuristics."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import List, Tuple


EMAIL_RE = re.compile(r"^[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}$", re.IGNORECASE)
PHONE_RE = re.compile(r"^\+?\d[\d\s\-().]{7,}\d$")
DATE_RE = re.compile(r"^\d{2}\.\d{2}\.\d{4}$|^\d{2}\.\d{4}$|^\d{4}-\d{2}-\d{2}$")
ZIP_RE = re.compile(r"^\d{5}$")

SECTION_MAP = {
    "profil": "SUMMARY",
    "kurzprofil": "SUMMARY",
    "über": "SUMMARY",
    "summary": "SUMMARY",
    "berufserfahrung": "WORK_EXPERIENCE",
    "experience": "WORK_EXPERIENCE",
    "work": "WORK_EXPERIENCE",
    "ausbildung": "EDUCATION",
    "education": "EDUCATION",
    "bildung": "EDUCATION",
    "skills": "SKILLS",
    "kompetenzen": "SKILLS",
    "fähigkeiten": "SKILLS",
    "sprachen": "LANGUAGES",
    "languages": "LANGUAGES",
    "qualifikationen": "QUALIFICATIONS",
    "qualifications": "QUALIFICATIONS",
}

PROFESSION_KEYWORDS = [
    "pflegefachkraft",
    "intensivpfleger",
    "medizinische",
    "fachangestellte",
    "pharmazeutisch",
    "technische",
    "assistent",
    "notfallsanitäter",
    "assistenzarzt",
    "assistenzärztin",
    "facharzt",
    "fachärztin",
    "oberarzt",
    "oberärztin",
    "chirurgie",
    "anästhesie",
    "innere",
    "pädiatrie",
    "kardiologie",
    "neurologie",
]

SKILL_KEYWORDS = [
    "pflegeplanung",
    "wundversorgung",
    "medikamentengabe",
    "vitalzeichenkontrolle",
    "hygienestandards",
    "palliativpflege",
    "patientenkommunikation",
    "intensivpflege",
    "beatmungsmanagement",
    "notfallmanagement",
    "ekg",
    "blutentnahme",
    "laborassistenz",
    "medizintechnik",
    "rehabilitation",
    "op",
    "anästhesie",
    "arzneimittel",
    "impfmanagement",
    "sterilgut",
    "dokumentation",
]

EDU_KEYWORDS = [
    "b.sc.",
    "m.sc.",
    "ausbildung",
    "hochschule",
    "universität",
    "akademie",
    "berufsschule",
    "berufskolleg",
    "pflegewissenschaft",
    "gesundheitsmanagement",
]

QUAL_KEYWORDS = [
    "führerschein",
    "hygieneschulung",
    "wundversorgung",
    "reanimation",
    "bls",
    "aed",
    "medikamentenmanagement",
    "strahlenschutz",
    "erstehilfe",
]

ORG_KEYWORDS = [
    "klinik",
    "krankenhaus",
    "universität",
    "hochschule",
    "akademie",
    "schule",
    "berufsschule",
    "berufskolleg",
]

TITLE_KEYWORDS = [
    "manager",
    "assistenzarzt",
    "assistenzärztin",
    "facharzt",
    "fachärztin",
    "oberarzt",
    "oberärztin",
    "pflegefachkraft",
    "intensivpfleger",
    "mta",
    "pta",
    "mfa",
]


def _norm(text: str) -> str:
    return re.sub(r"[^a-zA-Z0-9äöüÄÖÜß]+", "", text.lower())


def _label_span(labels: List[str], start: int, end: int, prefix: str) -> None:
    if start < 0 or end <= start:
        return
    labels[start] = f"B-{prefix}"
    for i in range(start + 1, end):
        labels[i] = f"I-{prefix}"


def _label_record(record: dict) -> int:
    tokens = record.get("tokens") or []
    if not tokens:
        record["labels"] = []
        return 0

    labels = ["O"] * len(tokens)
    labeled = 0

    # Name (first two tokens)
    if len(tokens) >= 2:
        _label_span(labels, 0, 1, "NAME_FIRST")
        _label_span(labels, 1, 2, "NAME_LAST")
        labeled += 2

    # Email / Phone (single token)
    for i, tok in enumerate(tokens):
        t = tok.strip()
        if EMAIL_RE.match(t):
            _label_span(labels, i, i + 1, "EMAIL")
            labeled += 1
        if PHONE_RE.match(t):
            _label_span(labels, i, i + 1, "PHONE")
            labeled += 1

    # Phone (multi-token): combine up to 4 tokens
    for i in range(len(tokens)):
        for span in range(2, 5):
            if i + span > len(tokens):
                continue
            candidate = " ".join(tokens[i:i+span]).strip()
            if PHONE_RE.match(candidate):
                _label_span(labels, i, i + span, "PHONE")
                labeled += span

    # Section headers (single-token heuristics)
    for i, tok in enumerate(tokens):
        key = _norm(tok)
        if key in SECTION_MAP:
            _label_span(labels, i, i + 1, SECTION_MAP[key])
            labeled += 1

    # Dates (generic)
    for i, tok in enumerate(tokens):
        if DATE_RE.match(tok):
            _label_span(labels, i, i + 1, "DATE")
            labeled += 1

    # Profession / Skill / Education keyword matching
    for i, tok in enumerate(tokens):
        key = _norm(tok)
        if any(k in key for k in PROFESSION_KEYWORDS):
            _label_span(labels, i, i + 1, "PROFESSION")
            labeled += 1
        if any(k in key for k in SKILL_KEYWORDS):
            _label_span(labels, i, i + 1, "SKILL")
            labeled += 1
        if any(k in key for k in EDU_KEYWORDS):
            _label_span(labels, i, i + 1, "EDU_KEYWORD")
            labeled += 1
        if any(k in key for k in QUAL_KEYWORDS):
            _label_span(labels, i, i + 1, "QUALIFICATION")
            labeled += 1
        if any(k in key for k in ORG_KEYWORDS):
            _label_span(labels, i, i + 1, "ORG")
            labeled += 1
        if any(k in key for k in TITLE_KEYWORDS):
            _label_span(labels, i, i + 1, "JOB_TITLE")
            labeled += 1

    # Section-aware spans: label tokens between section headers
    current_section = None
    for i, tok in enumerate(tokens):
        key = _norm(tok)
        if key in SECTION_MAP:
            current_section = SECTION_MAP[key]
            continue
        if current_section == "WORK_EXPERIENCE" and labels[i] == "O":
            _label_span(labels, i, i + 1, "WORK_TEXT")
            labeled += 1
        if current_section == "EDUCATION" and labels[i] == "O":
            _label_span(labels, i, i + 1, "EDU_TEXT")
            labeled += 1
        if current_section == "SKILLS" and labels[i] == "O":
            _label_span(labels, i, i + 1, "SKILL")
            labeled += 1
        if current_section == "QUALIFICATIONS" and labels[i] == "O":
            _label_span(labels, i, i + 1, "QUALIFICATION")
            labeled += 1

    # Zip/City
    for i, tok in enumerate(tokens):
        if ZIP_RE.match(tok):
            _label_span(labels, i, i + 1, "ZIPCODE")
            labeled += 1
            if i + 1 < len(tokens):
                _label_span(labels, i + 1, i + 2, "CITY")
                labeled += 1
            break

    record["labels"] = labels
    return labeled


def main() -> None:
    parser = argparse.ArgumentParser(description="Auto-label CV LayoutLM JSONL.")
    parser.add_argument("--input", type=Path, required=True, help="Input JSONL")
    parser.add_argument("--output", type=Path, required=True, help="Output JSONL")
    args = parser.parse_args()

    labeled_tokens = 0
    total_tokens = 0
    records = 0

    with args.input.open("r", encoding="utf-8") as f, args.output.open("w", encoding="utf-8") as out:
        for line in f:
            if not line.strip():
                continue
            record = json.loads(line)
            labeled_tokens += _label_record(record)
            total_tokens += len(record.get("tokens") or [])
            out.write(json.dumps(record, ensure_ascii=False) + "\n")
            records += 1

    coverage = (labeled_tokens / total_tokens) if total_tokens else 0.0
    print(f"Labeled {labeled_tokens}/{total_tokens} tokens ({coverage:.2%}) across {records} records.")


if __name__ == "__main__":
    main()
