import io

from PIL import Image

import caesar_ocr.ocr.engine as ocr
from caesar_ocr.ocr.postprocess import normalize_text, normalize_tokens


def _png_bytes(size=(10, 10), color=(255, 255, 255)) -> bytes:
    image = Image.new("RGB", size, color)
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    return buf.getvalue()


def test_detect_mrz_lines():
    lines = ["hello", "P<UTOERIKSSON<<ANNA<MARIA<<<<<<<<<<<<<<<<<<<", "foo<<bar<<baz"]
    mrz = ocr.detect_mrz_lines(lines)
    assert len(mrz) == 2


def test_classify_doc_mrz_overrides():
    predictions = [
        "P<UTOERIKSSON<<ANNA<MARIA<<<<<<<<<<<<<<<<<<<",
        "L898902C36UTO7408122F1204159ZE184226B<<<<<10",
    ]
    assert ocr.classify_doc(predictions) == "Passport"


def test_classify_doc_hints():
    assert ocr.classify_doc(["diploma"]) == "Degree Certificate"
    assert ocr.classify_doc(["invoice"]) == "Financial Report"


def test_extract_passport_fields_from_mrz():
    predictions = [
        "P<UTOERIKSSON<<ANNA<MARIA<<<<<<<<<<<<<<<<<<<",
        "L898902C36UTO7408122F1204159ZE184226B<<<<<10",
    ]
    fields = ocr.extract_passport_fields(predictions)
    assert fields.get("passport_number") == "L898902C3"
    assert fields.get("issuing_country") == "UTO"
    assert fields.get("surname") == "ERIKSSON"


def test_extract_passport_fields_fallback():
    predictions = ["passport no: X1234567"]
    fields = ocr.extract_passport_fields(predictions)
    assert fields.get("passport_number") == "X1234567"


def test_extract_passport_fields_from_ocr_text_lines():
    predictions = ["passport", "name", "foo"]
    ocr_text = (
        "Passport Name: Anna Example\n"
        "P<UTOERIKSSON<<ANNA<MARIA<<<<<<<<<<<<<<<<<<<\n"
        "L898902C36UTO7408122F1204159ZE184226B<<<<<10\n"
    )
    fields = ocr.extract_passport_fields(predictions, ocr_text=ocr_text)
    assert fields.get("passport_number") == "L898902C3"


def test_extract_financial_report_fields():
    text = (
        "Invoice No: ABC-123\n"
        "Accounting Period: 01/2024\n"
        "Customer: Jane Doe\n"
        "Total 1,234.56\n"
        "Date 2024-05-01"
    )
    fields = ocr.extract_financial_report_fields(text)
    assert "ABC-123" in fields.get("invoice_numbers", [])
    assert fields.get("accounting_period") == "01/2024"
    assert fields.get("customer_name_guess") == "Jane Doe"
    assert "2024-05-01" in fields.get("dates_detected", [])


def test_extract_diploma_fields():
    text = (
        "Technical University\n"
        "Name: John Doe\n"
        "Bachelor of Science\n"
        "Certified copy"
    )
    fields = ocr.extract_diploma_fields(text)
    assert fields.get("institution_guess") == "Technical University"
    assert fields.get("holder_name_guess") == "John Doe"
    assert fields.get("degree_type_guess") == "Bachelor"
    assert fields.get("is_certified_copy_hint") is True


def test_analyze_bytes_pdf(monkeypatch):
    dummy_image = Image.new("RGB", (10, 10), (255, 255, 255))

    def fake_load_images(_bytes, dpi=300):
        assert dpi == 300
        return [type("P", (), {"image": dummy_image, "page": 1})()]

    def fake_preprocess(_im):
        return "preprocessed"

    def fake_tokens(_im, **_kwargs):
        tokens = [
            {"text": "P<UTOERIKSSON<<ANNA<MARIA<<<<<<<<<<<<<<<<<<<"},
            {"text": "L898902C36UTO7408122F1204159ZE184226B<<<<<10"},
        ]
        return "ignored", tokens

    monkeypatch.setattr(ocr, "load_images_from_bytes", fake_load_images)
    monkeypatch.setattr(ocr, "preprocess_image", fake_preprocess)
    monkeypatch.setattr(ocr, "_ocr_tokens", fake_tokens)

    result = ocr.analyze_bytes(b"%PDF-1.4\n...")
    assert result.doc_type == "Passport"
    assert "passport_number" in result.fields


def test_analyze_bytes_image(monkeypatch):
    def fake_preprocess(_im):
        return "preprocessed"

    def fake_tokens(_im, **_kwargs):
        tokens = [{"text": "invoice"}]
        return "Invoice No: A12\nAccounting period: 01/2024", tokens

    monkeypatch.setattr(ocr, "preprocess_image", fake_preprocess)
    monkeypatch.setattr(ocr, "_ocr_tokens", fake_tokens)

    result = ocr.analyze_bytes(_png_bytes())
    assert result.doc_type == "Financial Report"
    assert "A12" in result.fields.get("invoice_numbers", [])


def test_normalize_text():
    assert normalize_text("a   b\nc") == "a b c"


def test_normalize_tokens():
    tokens = [{"text": "  hello "}, {"text": ""}, {"text": " world"}]
    normalized = normalize_tokens(tokens)
    assert [t["text"] for t in normalized] == ["hello", "world"]


def test_analyze_bytes_passes_lang(monkeypatch):
    seen = {"pred": None, "text": None}

    def fake_preprocess(_im):
        return "preprocessed"

    def fake_tokens(_im, **_kwargs):
        seen["pred"] = _kwargs.get("lang")
        seen["text"] = _kwargs.get("lang")
        tokens = [{"text": "invoice"}]
        return "Invoice No: A12\nAccounting period: 01/2024", tokens

    monkeypatch.setattr(ocr, "preprocess_image", fake_preprocess)
    monkeypatch.setattr(ocr, "_ocr_tokens", fake_tokens)

    _ = ocr.analyze_bytes(_png_bytes(), lang="deu")
    assert seen["pred"] == "deu"
    assert seen["text"] == "deu"
