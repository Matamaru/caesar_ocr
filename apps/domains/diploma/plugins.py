"""Diploma field extraction helpers."""

from __future__ import annotations

import re
from typing import Dict, Any

PERSON_NAME_RE = re.compile(r"(name|inhaber|inhaberin|holder|graduate)[:\s]+([A-ZÄÖÜ][^\n,;]{2,70})", re.I)
DEGREE_RE = re.compile(r"(Urkunde|Diplom|Bachelor|Master|Magister|Staatsexamen|Doctor|Doktor|PhD)", re.I)
DATE_RE = re.compile(
    r"(?:(?:19|20)\d{2}[-./](?:0?[1-9]|1[0-2])[-./](?:0?[1-9]|[12]\d|3[01]))|"
    r"(?:(?:0?[1-9]|[12]\d|3[01])[-./](?:0?[1-9]|1[0-2])[-./](?:19|20)\d{2})"
)


def extract_diploma_fields(ocr_text: str) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    lines = ocr_text.splitlines()
    if lines:
        out["institution_guess"] = lines[0].strip()
    m = PERSON_NAME_RE.search(ocr_text)
    if m:
        out["holder_name_guess"] = m.group(2).strip()
    m = DEGREE_RE.search(ocr_text)
    if m:
        out["degree_type_guess"] = m.group(1).strip()
    dates = DATE_RE.findall(ocr_text)
    if dates:
        out["dates_detected"] = dates
    if re.search(r"(certified copy|beglaubigte kopie|beglaubigung|copy)", ocr_text, re.I):
        out["is_certified_copy_hint"] = True
    return out
