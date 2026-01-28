"""Passport MRZ parsing helpers."""

from __future__ import annotations

from typing import Dict, Any, List, Tuple

_MRZ_WEIGHTS = [7, 3, 1]
_MRZ_MAP = {**{str(i): i for i in range(10)}, **{chr(ord('A') + i): 10 + i for i in range(26)}, "<": 0}


def _mrz_check_digit(value: str) -> str:
    total = 0
    for i, ch in enumerate(value):
        total += _MRZ_MAP.get(ch, 0) * _MRZ_WEIGHTS[i % 3]
    return str(total % 10)


def parse_td3_mrz(lines: List[str]) -> Dict[str, Any]:
    """Parse a TD3 passport MRZ (2 lines, 44 chars)."""
    out: Dict[str, Any] = {}
    if len(lines) != 2:
        return out

    l1 = lines[0].replace(" ", "").upper()
    l2 = lines[1].replace(" ", "").upper()
    if len(l1) < 30 or len(l2) < 30:
        return out

    try:
        out["document_code"] = l1[0:2]
        out["issuing_country"] = l1[2:5]
        name_raw = l1[5:].split("<<", 1)
        out["surname"] = name_raw[0].replace("<", " ").strip()
        out["given_names"] = name_raw[1].replace("<", " ").strip() if len(name_raw) > 1 else ""
        out["passport_number"] = l2[0:9].replace("<", "").strip()
        out["passport_number_check"] = l2[9:10]
        out["nationality"] = l2[10:13]
        out["birth_date_raw"] = l2[13:19]
        out["birth_date_check"] = l2[19:20]
        out["sex"] = l2[20:21]
        out["expiry_date_raw"] = l2[21:27]
        out["expiry_date_check"] = l2[27:28]
        out["personal_number"] = l2[28:42].replace("<", "").strip()
        out["personal_number_check"] = l2[42:43]
        out["final_check"] = l2[43:44]
        out["mrz_line1"] = l1
        out["mrz_line2"] = l2

        # Validate check digits where possible
        out["passport_number_valid"] = _mrz_check_digit(l2[0:9]) == out["passport_number_check"]
        out["birth_date_valid"] = _mrz_check_digit(l2[13:19]) == out["birth_date_check"]
        out["expiry_date_valid"] = _mrz_check_digit(l2[21:27]) == out["expiry_date_check"]
        out["personal_number_valid"] = _mrz_check_digit(l2[28:42]) == out["personal_number_check"]
        composite = l2[0:10] + l2[13:20] + l2[21:43]
        out["final_check_valid"] = _mrz_check_digit(composite) == out["final_check"]
    except Exception:
        return out
    return out


def detect_mrz_lines(text: str) -> List[str]:
    """Detect MRZ lines in OCR text (TD1/TD2/TD3)."""
    lines = [l.strip().replace(" ", "").upper() for l in text.splitlines() if l.strip()]
    mrz = []
    for line in lines:
        if all(ch.isalnum() or ch == "<" for ch in line) and len(line) >= 30:
            mrz.append(line)
    return mrz


def classify_mrz(lines: List[str]) -> str:
    """Return MRZ type: TD1, TD2, TD3, or unknown."""
    if len(lines) == 3 and all(len(l) == 30 for l in lines):
        return "TD1"
    if len(lines) == 2 and all(len(l) == 36 for l in lines):
        return "TD2"
    if len(lines) == 2 and all(len(l) == 44 for l in lines):
        return "TD3"
    return "unknown"


def validate_td3(lines: List[str]) -> Dict[str, Any]:
    """Validate TD3 MRZ check digits."""
    out = parse_td3_mrz(lines)
    if not out:
        return out
    out["mrz_type"] = "TD3"
    return out


def validate_td1(lines: List[str]) -> Dict[str, Any]:
    """Validate TD1 MRZ (ID cards)."""
    out: Dict[str, Any] = {"mrz_type": "TD1"}
    if len(lines) != 3 or any(len(l) != 30 for l in lines):
        return {}
    l1, l2, l3 = lines
    out["document_number"] = l1[5:14].replace("<", "").strip()
    out["document_number_check"] = l1[14:15]
    out["birth_date_raw"] = l2[0:6]
    out["birth_date_check"] = l2[6:7]
    out["sex"] = l2[7:8]
    out["expiry_date_raw"] = l2[8:14]
    out["expiry_date_check"] = l2[14:15]
    out["nationality"] = l2[15:18]
    out["document_number_valid"] = _mrz_check_digit(l1[5:14]) == out["document_number_check"]
    out["birth_date_valid"] = _mrz_check_digit(l2[0:6]) == out["birth_date_check"]
    out["expiry_date_valid"] = _mrz_check_digit(l2[8:14]) == out["expiry_date_check"]
    composite = l1[5:30] + l2[0:7] + l2[8:15] + l2[18:29]
    out["final_check"] = l2[29:30]
    out["final_check_valid"] = _mrz_check_digit(composite) == out["final_check"]
    out["mrz_line1"] = l1
    out["mrz_line2"] = l2
    out["mrz_line3"] = l3
    return out


def validate_td2(lines: List[str]) -> Dict[str, Any]:
    """Validate TD2 MRZ (visas/ID)."""
    out: Dict[str, Any] = {"mrz_type": "TD2"}
    if len(lines) != 2 or any(len(l) != 36 for l in lines):
        return {}
    l1, l2 = lines
    out["document_number"] = l2[0:9].replace("<", "").strip()
    out["document_number_check"] = l2[9:10]
    out["nationality"] = l2[10:13]
    out["birth_date_raw"] = l2[13:19]
    out["birth_date_check"] = l2[19:20]
    out["sex"] = l2[20:21]
    out["expiry_date_raw"] = l2[21:27]
    out["expiry_date_check"] = l2[27:28]
    out["document_number_valid"] = _mrz_check_digit(l2[0:9]) == out["document_number_check"]
    out["birth_date_valid"] = _mrz_check_digit(l2[13:19]) == out["birth_date_check"]
    out["expiry_date_valid"] = _mrz_check_digit(l2[21:27]) == out["expiry_date_check"]
    out["mrz_line1"] = l1
    out["mrz_line2"] = l2
    return out


def infer_mrz(text: str) -> Dict[str, Any]:
    """Detect and validate MRZ lines from OCR text."""
    lines = detect_mrz_lines(text)
    mrz_type = classify_mrz(lines)
    if mrz_type == "TD3":
        return validate_td3(lines[:2])
    if mrz_type == "TD2":
        return validate_td2(lines[:2])
    if mrz_type == "TD1":
        return validate_td1(lines[:3])
    return {}
