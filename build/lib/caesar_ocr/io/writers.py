"""Output writers (JSON, CSV)."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List


def write_json(path: Path, data: Any, *, indent: int = 2) -> None:
    """
    Write data to a JSON file.

    :param path: Path to the output JSON file.
    :param data: Data to write as JSON.
    :param indent: Number of spaces for indentation in the JSON file.
    """
    path.parent.mkdir(parents=True, exist_ok=True) # Ensure the parent directory exists
    path.write_text(json.dumps(data, ensure_ascii=False, indent=indent)) # Write JSON data to file


def write_jsonl(path: Path, rows: Iterable[Dict[str, Any]]) -> None:
    """
    Write an iterable of dictionaries to a JSONL file.
    JSONL (JSON Lines) format has one JSON object per line.
    
    :param path: Path to the output JSONL file.
    :param rows: Iterable of dictionaries to write as JSON lines.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_csv(path: Path, rows: List[Dict[str, Any]], *, fieldnames: List[str]) -> None:
    """
    Write a list of dictionaries to a CSV file.

    :param path: Path to the output CSV file.
    :param rows: List of dictionaries to write as CSV rows.
    :param fieldnames: List of field names (column headers) for the CSV.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def flatten_fields_to_rows(data: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Flatten schema output into CSV rows for fields/entities.
    
    :param data: OCR data containing fields/entities.
    :return: List of dictionaries representing CSV rows.
    """
    # Extract OCR information
    ocr = data.get("ocr", {})
    pages = ocr.get("pages", [])
    fields = ocr.get("fields", {})
    doc_type = ocr.get("doc_type")
    language = ocr.get("language")

    # Prepare rows for CSV
    rows: List[Dict[str, Any]] = []
    if not fields:
        rows.append({"doc_type": doc_type, "language": language})
        return rows

    # Flatten each field into a row
    for key, value in fields.items():
        rows.append(
            {
                "doc_type": doc_type,
                "language": language,
                "field": key,                   # Field name, e.g., "total_amount"  
                "value": value,                 # Field value, e.g., "123.45"
                "pages": len(pages),
            }
        )
    return rows


def tokens_to_rows(data: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Flatten OCR tokens into CSV rows (one row per token).

    :param data: OCR data containing tokens.
    :return: List of dictionaries representing CSV rows.
    """
    # Extract OCR information
    ocr = data.get("ocr", {})
    pages = ocr.get("pages", [])
    doc_type = ocr.get("doc_type")
    language = ocr.get("language")

    # Prepare rows for CSV
    rows: List[Dict[str, Any]] = []

    # Flatten each token into a row per page
    for page in pages:
        page_num = page.get("page")
        for token in page.get("tokens", []):
            rows.append(
                {
                    "doc_type": doc_type,                       # Document type, e. g., "invoice"
                    "language": language,                       # Language code, e. g., "en" or "de"
                    "page": page_num,                           # Page number
                    "text": token.get("text"),                  # Token text
                    "bbox": token.get("bbox"),                  # Bounding box coordinates
                    "start": token.get("start"),                # Character offsets
                    "end": token.get("end"),                    # Character offsets
                    "conf": token.get("conf"),                  # Confidence score
                    "label": token.get("label"),                # LayoutLM label
                    "label_score": token.get("label_score"),    # Label confidence score
                }
            )
    return rows


def token_labels_by_page_rows(data: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Summarize LayoutLM token labels grouped by page.

    :param data: OCR data containing tokens with labels.
    :return: List of dictionaries representing summarized label counts by page.
    """
    ocr = data.get("ocr", {})
    pages = ocr.get("pages", [])
    doc_type = ocr.get("doc_type")
    language = ocr.get("language")

    rows: List[Dict[str, Any]] = []
    for page in pages:
        page_num = page.get("page")
        labels = [t.get("label") for t in page.get("tokens", []) if t.get("label")]
        counts: Dict[str, int] = {}
        for label in labels:
            counts[label] = counts.get(label, 0) + 1
        rows.append(
            {
                "doc_type": doc_type,
                "language": language,
                "page": page_num,
                "label_counts": counts,
            }
        )
    return rows
