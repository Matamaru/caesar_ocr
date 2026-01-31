"""OCR pipeline for PDFs/images with lightweight heuristics.

This module provides a single entry point, `analyze_bytes`, which:
- loads an image or first PDF page,
- runs OCR with Tesseract,
- classifies the document with simple keyword heuristics,
- extracts a small set of fields for known document types.
"""

#=== Imports =============================================================
import io
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence

from PIL import Image

from .postprocess import preprocess_image, normalize_text, normalize_tokens
from .tesseract import ocr_predictions, ocr_text, ocr_tokens
from ..io.loaders import load_images_from_bytes

#=== Helpers =============================================================

def _load_image_from_bytes(b: bytes) -> Image.Image:
    """Load image bytes and normalize to RGB (or L) PIL image."""
    im = Image.open(io.BytesIO(b))
    if im.mode not in ("RGB", "L"):  # normalize
        im = im.convert("RGB")
    return im


def _ocr_predictions(preprocessed_im, lang: str = "eng+deu", psm: int = 11) -> List[str]:
    """Compatibility wrapper for tests; delegates to tesseract adapter."""
    return ocr_predictions(preprocessed_im, lang=lang, psm=psm)


def _ocr_text(preprocessed_im, lang: str = "eng+deu", psm: int = 3) -> str:
    """Compatibility wrapper for tests; delegates to tesseract adapter."""
    return ocr_text(preprocessed_im, lang=lang, psm=psm)


def _ocr_tokens(preprocessed_im, lang: str = "eng+deu", psm: int = 6) -> tuple[str, List[Dict[str, Any]]]:
    """Compatibility wrapper for tests; delegates to tesseract adapter."""
    return ocr_tokens(preprocessed_im, lang=lang, psm=psm)


def detect_mrz_lines(all_predictions: List[str]) -> List[str]:
    """Return candidate MRZ lines from token-level predictions."""
    mrz_lines = []

    # MRZ lines are dense with '<' fillers; 3+ is a cheap signal.
    for line in all_predictions:
        if line.count("<") >= 3:
            mrz_lines.append(line)

    return mrz_lines


def detect_mrz_lines_from_text(ocr_text: str) -> List[str]:
    """Return candidate MRZ lines from OCR text lines."""
    if not ocr_text:
        return []
    mrz_lines = []
    for line in ocr_text.splitlines():
        if "<" not in line:
            continue
        cleaned = re.sub(r"[^A-Z0-9<]+", "", line.upper())
        if "P<" in cleaned:
            start = cleaned.find("P<")
            if len(cleaned) >= start + 44:
                mrz_lines.append(cleaned[start : start + 44])
                # If we can grab a second line immediately after, do so.
                if len(cleaned) >= start + 88:
                    mrz_lines.append(cleaned[start + 44 : start + 88])
                else:
                    # Try to find a valid MRZ line 2 anywhere in the cleaned text.
                    line2 = _find_mrz_line2(cleaned)
                    if line2:
                        mrz_lines.append(line2)
        if mrz_lines:
            return mrz_lines
    if mrz_lines:
        return mrz_lines
    # Fallback: extract MRZ-like chunks from single-line OCR text.
    cleaned = re.sub(r"[^A-Z0-9<]+", "", ocr_text.upper())
    if "P<" in cleaned:
        start = cleaned.find("P<")
        if len(cleaned) >= start + 44:
            mrz_lines.append(cleaned[start : start + 44])
            if len(cleaned) >= start + 88:
                mrz_lines.append(cleaned[start + 44 : start + 88])
            else:
                line2 = _find_mrz_line2(cleaned)
                if line2:
                    mrz_lines.append(line2)
    if not mrz_lines:
        line2 = _find_mrz_line2(cleaned)
        if line2:
            mrz_lines.append(line2)
    return mrz_lines


def _find_mrz_line2(text: str) -> str | None:
    """Find a TD3 MRZ line 2 candidate inside text."""
    pattern = re.compile(
        r"[A-Z0-9<]{9}[0-9][A-Z]{3}[0-9]{6}[0-9][MF<][0-9]{6}[0-9][A-Z0-9<]{14}[0-9][0-9]"
    )
    match = pattern.search(text)
    if match:
        return match.group(0)
    return None


#=== Doc classifiers (very light heuristics) =============================

# Hints for document classification
PASSPORT_HINTS = {"passport", "reisepass", "passeport", "passport no", "passnummer", "staat", "nationality"}
DIPLOMA_HINTS_DE = {"zeugnis", "hochschule", "universität", "fachhochschule", "abschluss", "urkunde", "diplom"}
DIPLOMA_HINTS_EN = {"diploma", "degree", "university", "college", "certificate", "transcript"}  # transcript only as hint
FINANCIAL_REPORT_HINTS = {
    "invoice",
    "invoice no",
    "invoice number",
    "invoice date",
    "accounting period",
    "amount",
    "total",
    "balance",
    "customer",
}

#=== Document classification =============================================

def classify_doc(predictions: List[str], *, ocr_text: str | None = None) -> str:
    """Classify document type using simple keyword heuristics."""
    # MRZ implies a passport-like document.
    mrz_lines = detect_mrz_lines_from_text(ocr_text or "") if ocr_text else []
    if not mrz_lines:
        mrz_lines = detect_mrz_lines(predictions)
    if len(mrz_lines) > 0:
        return "Passport"
    if any(h in predictions for h in PASSPORT_HINTS):
        return "Passport"
    if any(h in predictions for h in DIPLOMA_HINTS_DE | DIPLOMA_HINTS_EN):
        return "Degree Certificate"
    if any(h in predictions for h in FINANCIAL_REPORT_HINTS):
        return "Financial Report"
    return "unknown"

#=== Field extraction ==================================================

# Date pattern: YYYY-MM-DD or DD-MM-YYYY with - . / separators.
DATE_RE = re.compile(r"(?:(?:19|20)\d{2}[-./](?:0?[1-9]|1[0-2])[-./](?:0?[1-9]|[12]\d|3[01]))|"
                     r"(?:(?:0?[1-9]|[12]\d|3[01])[-./](?:0?[1-9]|1[0-2])[-./](?:19|20)\d{2})")

# MRZ TD3 parser (2 lines, 44 chars each) – tolerant cleanup


def _extract_passport_data_from_mrz(mrz_lines: List[str]) -> Dict[str, Any]:
    """Parse TD3-style MRZ (2 lines) into a minimal field dict."""
    out: Dict[str, Any] = {}

    # TD3 layout:
    # L1: P<CCNAME<<GIVEN<<<<<<<<<<<<<<<<<<<<<<<<
    # L2: PASSPORTNO<CHECK>CCYYMMDD<CHECK>SEX EXP<CHK>NatID<CHK> <<optional
    if len(mrz_lines) >= 2:
        # Clean lines and uppercase
        l1 = mrz_lines[0].replace(" ", "").upper()
        l2 = mrz_lines[1].replace(" ", "").upper()

        # Add to output
        out = {"mrz_line1": l1, "mrz_line2": l2}

        # Parse fields (best effort; OCR may be noisy).
        try:
            # Extract fields based on fixed positions
            out["document_code"] = l1[0:2]
            out["issuing_country"] = l1[2:5]
            name_raw = l1[5:].split("<<", 1)
            out["surname"] = name_raw[0].replace("<", " ").strip()
            out["given_names"] = name_raw[1].replace("<", " ").strip() if len(name_raw) > 1 else ""
            out["passport_number"] = l2[0:9].replace("<", "").strip()
            out["nationality"] = l2[10:13]
            out["birth_date_raw"] = l2[13:19]  # YYMMDD
            out["sex"] = l2[20:21]
            out["expiry_date_raw"] = l2[21:27]  # YYMMDD
        except Exception:
            # Keep partial output; MRZ OCR is often imperfect.
            pass
    return out


def extract_passport_fields(predictions: List[str], *, ocr_text: str | None = None) -> Dict[str, Any]:
    """Extract passport-like fields from OCR tokens and/or raw OCR text."""
    text_for_mrz = ocr_text or ""
    mrz_lines = detect_mrz_lines_from_text(text_for_mrz)
    if not mrz_lines:
        mrz_lines = detect_mrz_lines(predictions)
    passport_data = _extract_passport_data_from_mrz(mrz_lines)
    ocr_text = text_for_mrz or "\n".join(predictions)
    # Fallback: look for a labeled passport number in body text.
    if "passport_number" not in passport_data:
        m = re.search(r"(passport|passnummer|passport no)\s*[:\-]?\s*([A-Z0-9]{6,})", ocr_text, re.I)
        if m:
            passport_data["passport_number"] = m.group(2)
    return passport_data

# Diploma field extraction
# Person name pattern: look for labels like "Name:", "Inhaber:", etc.
PERSON_NAME_RE = re.compile(r"(name|inhaber|inhaberin|holder|graduate)[:\s]+([A-ZÄÖÜ][^\n,;]{2,70})", re.I)
# Degree type pattern
DEGREE_RE = re.compile(r"(Urkunde|Diplom|Bachelor|Master|Magister|Staatsexamen|Doctor|Doktor|PhD)", re.I)
DEGREE_LINE_RE = re.compile(r"\b(Diploma|Bachelor|Master|Doctor|PhD)\s+Degree\b", re.I)
DEGREE_OF_RE = re.compile(r"\b(Bachelor|Master|Doctor|PhD)\s+of\b", re.I)
PROGRAM_RE = re.compile(r"\b(Program|Studiengang)\s*[:\-]\s*(.+)", re.I)
INSTITUTION_RE = re.compile(r"\b(University|Hochschule)\s*[:\-]\s*(.+)", re.I)
INSTITUTION_OF_RE = re.compile(r"\b((?:University of|Universitaet|Universität|Hochschule)\s+.+)", re.I)
INSTITUTION_LABEL_RE = re.compile(r"\b(Hochschule|Universitaet|Universität|University)\s*[:\-]\s*(.+)", re.I)
LOCATION_RE = re.compile(r"\b(Location|Ort)\s*[:\-]\s*(.+)", re.I)
DATE_LABEL_RE = re.compile(r"\b(Date|Datum)\s*[:\-]\s*(\d{2}\.\d{2}\.\d{4})", re.I)
DIPLOMA_NO_RE = re.compile(r"\b(Diploma No\.|Urkunden-Nr\.)\s*[:#\-]?\s*([A-Z0-9\-]+)", re.I)
AWARDED_RE = re.compile(r"\b(awarded to|verliehen an)\s+(.+)", re.I)


def _capture_until_label(
    text: str,
    pattern: re.Pattern,
    *,
    stop_labels: List[str],
    group_index: int = 2,
) -> Optional[str]:
    match = pattern.search(text)
    if not match:
        return None
    value = match.group(group_index).strip()
    if not value:
        return None
    stop_re = re.compile(r"\b(" + "|".join(stop_labels) + r")\b", re.I)
    stop_match = stop_re.search(value)
    if stop_match:
        value = value[: stop_match.start()].strip(" -|")
    value = value.split("|", 1)[0].strip(" -|")
    return value.strip(" -|") or None


def _split_trailing_name(value: str) -> tuple[str, Optional[str]]:
    match = re.match(r"(.+?)\s+([A-ZÄÖÜ][A-Za-zÄÖÜäöüß\\-]+\\s+[A-ZÄÖÜ][A-Za-zÄÖÜäöüß\\-]+)$", value)
    if not match:
        # Fallback: split last two capitalized words.
        parts = value.split()
        if len(parts) >= 4:
            last_two = parts[-2:]
            if all(p[:1].isalpha() and p[:1].upper() == p[:1] for p in last_two):
                return " ".join(parts[:-2]).strip(), " ".join(last_two).strip()
        return value, None
    return match.group(1).strip(), match.group(2).strip()

# Financial report field extraction
INVOICE_NO_RE = re.compile(
    r"\b(invoice|rechnung)\s*(no|number|nr)?\s*[:#\-]?\s*([A-Z0-9][A-Z0-9\-]{1,})",
    re.I,
)
AMOUNT_RE = re.compile(r"(?:^|\b)(\d{1,3}(?:[.,]\d{3})*(?:[.,]\d{2}))\b")
ACCOUNTING_PERIOD_RE = re.compile(r"(accounting\s*period|abrechnungszeitraum)[:\s]*([A-Z0-9./ \-]{3,})", re.I)
CUSTOMER_RE = re.compile(r"(customer|kunde|client)[:\s]*([A-ZÄÖÜ][^\n,;]{2,70})", re.I)


def extract_financial_report_fields(ocr_text: str) -> Dict[str, Any]:
    """Extract basic fields from invoice-like text (best-effort heuristics)."""
    out: Dict[str, Any] = {}

    # Accounting period is often explicitly labeled.
    m = ACCOUNTING_PERIOD_RE.search(ocr_text)
    if m:
        out["accounting_period"] = m.group(2).strip()

    # Collect invoice numbers (may appear multiple times).
    invoice_numbers = []
    for match in INVOICE_NO_RE.finditer(ocr_text):
        invoice_numbers.append(match.group(3).strip())
    if invoice_numbers:
        out["invoice_numbers"] = sorted(set(invoice_numbers))

    # Extract dates and amounts to help downstream logic.
    dates = DATE_RE.findall(ocr_text)
    if dates:
        out["dates_detected"] = dates

    amounts = []
    for match in AMOUNT_RE.finditer(ocr_text):
        amounts.append(match.group(1))
    if amounts:
        out["amounts_detected"] = amounts

    # Pick a single customer name if present.
    m = CUSTOMER_RE.search(ocr_text)
    if m:
        out["customer_name_guess"] = m.group(2).strip()

    return out


def extract_diploma_fields(ocr_text: str) -> Dict[str, Any]:
    """Extract diploma-like fields from OCR text."""
    out: Dict[str, Any] = {}
    text = " ".join(ocr_text.split())
    stop_labels = [
        "Program",
        "Studiengang",
        "Status",
        "Location",
        "Ort",
        "Date",
        "Datum",
        "Diploma No.",
        "Urkunden-Nr.",
        "Signature",
    ]
    stop_labels_institution = [label for label in stop_labels if label not in ("University", "Hochschule")]
    stop_labels_program = stop_labels + ["University", "Hochschule"]

    holder = _capture_until_label(text, AWARDED_RE, stop_labels=stop_labels)
    if holder:
        out["holder_name_guess"] = holder
    else:
        m = PERSON_NAME_RE.search(text)
        if m:
            out["holder_name_guess"] = m.group(2).strip()

    program = _capture_until_label(text, PROGRAM_RE, stop_labels=stop_labels_program)
    if program:
        out["program_guess"] = program

    institution = _capture_until_label(text, INSTITUTION_RE, stop_labels=stop_labels_institution)
    if institution:
        institution, trailing_name = _split_trailing_name(institution)
        out["institution_guess"] = institution
        if trailing_name and "holder_name_guess" not in out:
            out["holder_name_guess"] = trailing_name
    else:
        institution = _capture_until_label(text, INSTITUTION_LABEL_RE, stop_labels=stop_labels_institution)
        if institution:
            institution, trailing_name = _split_trailing_name(institution)
            out["institution_guess"] = institution
            if trailing_name and "holder_name_guess" not in out:
                out["holder_name_guess"] = trailing_name
        else:
            institution = _capture_until_label(
                text,
                INSTITUTION_OF_RE,
                stop_labels=stop_labels_institution,
                group_index=1,
            )
            if institution:
                institution, trailing_name = _split_trailing_name(institution)
                out["institution_guess"] = institution
                if trailing_name and "holder_name_guess" not in out:
                    out["holder_name_guess"] = trailing_name

    location = _capture_until_label(text, LOCATION_RE, stop_labels=stop_labels)
    if location:
        out["location_guess"] = location

    m = DATE_LABEL_RE.search(text)
    if m:
        out["issue_date_guess"] = m.group(2).strip()

    m = DIPLOMA_NO_RE.search(text)
    if m:
        out["diploma_number_guess"] = m.group(2).strip()

    m = DEGREE_LINE_RE.findall(text)
    if m:
        out["degree_type_guess"] = m[-1].strip()
    else:
        m = DEGREE_OF_RE.findall(text)
        if m:
            out["degree_type_guess"] = m[-1].strip()
        else:
            m = DEGREE_RE.findall(text)
            if m:
                out["degree_type_guess"] = m[-1].strip()
    if out.get("degree_type_guess") == "Urkunde" and "Diplom" in text:
        out["degree_type_guess"] = "Diplom"

    # Dates help identify issuance period (fallback for debugging).
    dates = DATE_RE.findall(text)
    if dates:
        out["dates_detected"] = dates

    # Certified copy hint is used as a lightweight flag.
    if re.search(r"(certified copy|beglaubigte kopie|beglaubigung|copy)", text, re.I):
        out["is_certified_copy_hint"] = True

    return out

#=== Public API ===================================================================

@dataclass
class OcrResult:
    """Result of OCR analysis.
    Contains document type, full text, extracted fields, word-level data.
    JSON-serializable.
    """
    doc_type: str
    predictions: List[str]
    ocr_text: str
    fields: Dict[str, Any]
    tokens: List[Dict[str, Any]]
    page_texts: List[str]


def _run_ocr(im: Image.Image, *, lang: str, page: int) -> Dict[str, Any]:
    pim = preprocess_image(im)
    ocr_text, tokens = _ocr_tokens(pim, lang=lang, psm=6)
    tokens = normalize_tokens(tokens)
    for token in tokens:
        token["page"] = page
    predictions = [t["text"].lower() for t in tokens]
    return {"ocr_text": normalize_text(ocr_text), "tokens": tokens, "predictions": predictions}


def analyze_pages(pages: Sequence, *, lang: str = "eng+deu") -> OcrResult:
    all_predictions: List[str] = []
    all_text: List[str] = []
    all_tokens: List[Dict[str, Any]] = []

    for page in pages:
        result = _run_ocr(page.image, lang=lang, page=page.page)
        all_predictions.extend(result["predictions"])
        all_text.append(result["ocr_text"])
        all_tokens.extend(result["tokens"])

    predictions = all_predictions
    ocr_text = "\n".join(all_text)
    tokens = all_tokens

    doc_type = classify_doc(predictions, ocr_text=ocr_text)
    fields: Dict[str, Any] = {}

    if doc_type == "Passport":
        fields = extract_passport_fields(predictions, ocr_text=ocr_text)
    elif doc_type == "Degree Certificate":
        fields = extract_diploma_fields(ocr_text)
    elif doc_type == "Financial Report":
        fields = extract_financial_report_fields(ocr_text)

    return OcrResult(
        doc_type=doc_type,
        predictions=predictions,
        ocr_text=ocr_text,
        fields=fields,
        tokens=tokens,
        page_texts=all_text,
    )


def analyze_bytes(file_bytes: bytes, *, lang: str = "eng+deu") -> OcrResult:
    """Run OCR and heuristic extraction on PDF/image bytes."""
    pages = load_images_from_bytes(file_bytes, dpi=300)
    return analyze_pages(pages, lang=lang)
