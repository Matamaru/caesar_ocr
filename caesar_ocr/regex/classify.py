"""Lightweight keyword-based document classification."""

from __future__ import annotations

import re
from typing import Dict, Iterable, List, Set


CANONICAL_DOCS: Dict[str, List[str]] = {
    "passport": ["passport", "pass", "reiseausweis", "reisepass", "passeport", "passnummer"],
    "id_card": ["id card", "personalausweis", "ausweis"],
    "diploma": ["diploma", "degree", "zeugnis", "urkunde", "abschluss", "hochschule", "universität"],
    "transcript": ["transcript", "marksheet", "course list", "leistungsnachweis"],
    "license": ["license", "registration", "approbation", "zulassung"],
    "birth_certificate": ["birth", "geburtsurkunde"],
    "cv": ["cv", "lebenslauf", "curriculum vitae"],
    "language_b2": ["b2", "sprachzertifikat"],
    "language_b2_pflege": ["b2 pflege", "pflege b2"],
    "good_standing": ["good standing", "gsc"],
    "apostille": ["apostille", "legalization", "legalisation"],
    "certified_translation": ["translation", "übersetzung", "uebersetzung", "beglaubigt"],
}


def infer_present_docs(text: str) -> Set[str]:
    """Infer which canonical document types are present based on OCR text."""
    present = set()
    lower_text = text.lower()
    for canon, keywords in CANONICAL_DOCS.items():
        for kw in keywords:
            if re.search(rf"\\b{re.escape(kw)}\\b", lower_text):
                present.add(canon)
                break
    return present

