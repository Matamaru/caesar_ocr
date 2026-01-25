import json
from pathlib import Path

from caesar_ocr.pipeline.analyze import analyze_document_bytes


def test_pipeline_schema_output(monkeypatch):
    def fake_analyze_bytes(_bytes, lang="eng+deu"):
        class Dummy:
            doc_type = "Financial Report"
            predictions = []
            ocr_text = "Invoice No: A12"
            fields = {"invoice_numbers": ["A12"]}
            tokens = [
                {
                    "text": "Invoice",
                    "bbox": [1, 2, 3, 4],
                    "start": 0,
                    "end": 7,
                    "conf": 0.9,
                    "page": 1,
                }
            ]
            page_texts = ["Invoice No: A12"]

        return Dummy()

    def fake_load_images(_bytes, dpi=300):
        return [type("P", (), {"page": 1, "width": 100, "height": 200, "image": object()})()]

    monkeypatch.setattr("caesar_ocr.pipeline.analyze.analyze_bytes", fake_analyze_bytes)
    monkeypatch.setattr("caesar_ocr.pipeline.analyze.load_images_from_bytes", fake_load_images)

    result = analyze_document_bytes(b"dummy")
    data = result.to_dict()

    assert data["ocr"]["doc_type"] == "Financial Report"
    assert data["ocr"]["language"] == "eng+deu"
    assert data["ocr"]["pages"][0]["width"] == 100
    assert data["ocr"]["fields"]["invoice_numbers"] == ["A12"]
    assert data["ocr"]["pages"][0]["tokens"][0]["text"] == "Invoice"
    assert data["ocr"]["pages"][0]["tokens"][0]["bbox"] == [1, 2, 3, 4]
    assert data["ocr"]["pages"][0]["tokens"][0]["start"] == 0
    assert data["ocr"]["pages"][0]["tokens"][0]["end"] == 7
    assert data["ocr"]["pages"][0]["tokens"][0]["conf"] == 0.9


def test_pipeline_regex_rules(tmp_path: Path, monkeypatch):
    rules_path = tmp_path / "rules.yaml"
    rules_path.write_text(
        """
- name: invoice_number
  pattern: '(?i)invoice\\s*(no|number)?\\s*[:#-]?\\s*([A-Z0-9-]{3,})'
  group: 2
  output_field: invoice_number
"""
    )

    def fake_analyze_bytes(_bytes, lang="eng+deu"):
        class Dummy:
            doc_type = "unknown"
            predictions = []
            ocr_text = "Invoice No: ABC-123"
            fields = {}
            tokens = []
            page_texts = ["Invoice No: ABC-123"]

        return Dummy()

    def fake_load_images(_bytes, dpi=300):
        return [type("P", (), {"page": 1, "width": 100, "height": 200, "image": object()})()]

    monkeypatch.setattr("caesar_ocr.pipeline.analyze.analyze_bytes", fake_analyze_bytes)
    monkeypatch.setattr("caesar_ocr.pipeline.analyze.load_images_from_bytes", fake_load_images)

    result = analyze_document_bytes(b"dummy", regex_rules_path=str(rules_path))
    data = result.to_dict()

    assert data["ocr"]["fields"]["invoice_number"] == "ABC-123"


def test_pipeline_multi_page_tokens(monkeypatch):
    def fake_analyze_bytes(_bytes, lang="eng+deu"):
        class Dummy:
            doc_type = "unknown"
            predictions = []
            ocr_text = "Page1\nPage2"
            fields = {}
            tokens = [
                {"text": "Hello", "bbox": [0, 0, 1, 1], "page": 1},
                {"text": "World", "bbox": [0, 0, 1, 1], "page": 2},
            ]
            page_texts = ["Page1", "Page2"]

        return Dummy()

    def fake_load_images(_bytes, dpi=300):
        return [
            type("P", (), {"page": 1, "width": 100, "height": 200, "image": object()})(),
            type("P", (), {"page": 2, "width": 100, "height": 200, "image": object()})(),
        ]

    monkeypatch.setattr("caesar_ocr.pipeline.analyze.analyze_bytes", fake_analyze_bytes)
    monkeypatch.setattr("caesar_ocr.pipeline.analyze.load_images_from_bytes", fake_load_images)

    result = analyze_document_bytes(b"dummy")
    data = result.to_dict()

    assert len(data["ocr"]["pages"]) == 2
    assert data["ocr"]["pages"][0]["text"] == "Page1"
    assert data["ocr"]["pages"][1]["text"] == "Page2"
    assert data["ocr"]["pages"][0]["tokens"][0]["text"] == "Hello"
    assert data["ocr"]["pages"][1]["tokens"][0]["text"] == "World"
