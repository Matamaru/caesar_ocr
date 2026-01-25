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
from typing import Any, Dict, List

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
    """Return candidate MRZ lines (heuristic: contains multiple '<' chars)."""
    mrz_lines = []

    # MRZ lines are dense with '<' fillers; 3+ is a cheap signal.
    for line in all_predictions:
        if line.count("<") >= 3:
            mrz_lines.append(line)

    return mrz_lines


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

def classify_doc(predictions: List[str]) -> str:
    """Classify document type using simple keyword heuristics."""
    # MRZ implies a passport-like document.
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
    if len(mrz_lines) == 2:
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


def extract_passport_fields(predictions: List[str]) -> Dict[str, Any]:
    """Extract passport-like fields from OCR predictions."""
    passport_data = _extract_passport_data_from_mrz(detect_mrz_lines(predictions))
    ocr_text = "\n".join(predictions)
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
    # Institution names are frequently in the first line.
    lines = ocr_text.splitlines()
    if lines:
        out["institution_guess"] = lines[0].strip()
    # Holder name and degree type are scanned with simple regexes.
    m = PERSON_NAME_RE.search(ocr_text)
    if m:
        out["holder_name_guess"] = m.group(2).strip()
    m = DEGREE_RE.search(ocr_text)
    if m:
        out["degree_type_guess"] = m.group(1).strip()
    # Dates help identify issuance period.
    dates = DATE_RE.findall(ocr_text)
    if dates:
        out["dates_detected"] = dates
    # Certified copy hint is used as a lightweight flag.
    if re.search(r"(certified copy|beglaubigte kopie|beglaubigung|copy)", ocr_text, re.I):
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


def analyze_bytes(file_bytes: bytes, *, lang: str = "eng+deu") -> OcrResult:
    """Run OCR and heuristic extraction on PDF/image bytes."""
    pages = load_images_from_bytes(file_bytes, dpi=300)

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

    doc_type = classify_doc(predictions)
    fields: Dict[str, Any] = {}

    # Field extraction depends on document type.
    if doc_type == "Passport":
        fields = extract_passport_fields(predictions)
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
