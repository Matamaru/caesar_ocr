import json
from pathlib import Path

import caesar_ocr.cli as cli


def test_cli_outputs_json(monkeypatch, tmp_path, capsys):
    def fake_analyze(
        _bytes,
        layoutlm_model_dir=None,
        lang="eng+deu",
        layoutlm_lang=None,
        regex_rules_path=None,
        regex_debug=False,
        layoutlm_token_model_dir=None,
    ):
        class Dummy:
            def to_dict(self, schema: bool = True):
                return {"ok": True, "lang": lang}

        return Dummy()

    sample = tmp_path / "doc.pdf"
    sample.write_bytes(b"%PDF-1.4")

    monkeypatch.setattr(cli, "analyze_document_bytes", fake_analyze)
    monkeypatch.setattr("sys.argv", ["caesar-ocr", "analyze", str(sample)])

    cli.main()
    out = capsys.readouterr().out.strip()
    data = json.loads(out)
    assert data["ok"] is True
    assert data["lang"] == "eng+deu"


def test_cli_writes_output_file(monkeypatch, tmp_path):
    def fake_analyze(
        _bytes,
        layoutlm_model_dir=None,
        lang="eng+deu",
        layoutlm_lang=None,
        regex_rules_path=None,
        regex_debug=False,
        layoutlm_token_model_dir=None,
    ):
        class Dummy:
            def to_dict(self, schema: bool = True):
                return {"ok": True}

        return Dummy()

    sample = tmp_path / "doc.pdf"
    sample.write_bytes(b"%PDF-1.4")
    out_path = tmp_path / "out.json"

    monkeypatch.setattr(cli, "analyze_document_bytes", fake_analyze)
    monkeypatch.setattr("sys.argv", ["caesar-ocr", "analyze", str(sample), "--output", str(out_path)])

    cli.main()
    assert json.loads(out_path.read_text()) == {"ok": True}


def test_cli_csv_fields(monkeypatch, tmp_path):
    def fake_analyze(
        _bytes,
        layoutlm_model_dir=None,
        lang="eng+deu",
        layoutlm_lang=None,
        regex_rules_path=None,
        regex_debug=False,
        layoutlm_token_model_dir=None,
    ):
        class Dummy:
            def to_dict(self, schema: bool = True):
                return {
                    "ocr": {
                        "doc_type": "Financial Report",
                        "language": "eng+deu",
                        "fields": {"invoice_number": "ABC-123"},
                        "pages": [],
                    }
                }

        return Dummy()

    sample = tmp_path / "doc.pdf"
    sample.write_bytes(b"%PDF-1.4")
    csv_path = tmp_path / "fields.csv"

    monkeypatch.setattr(cli, "analyze_document_bytes", fake_analyze)
    monkeypatch.setattr("sys.argv", ["caesar-ocr", "analyze", str(sample), "--csv-fields", str(csv_path)])

    cli.main()
    text = csv_path.read_text().splitlines()
    assert "field" in text[0]
    assert "invoice_number" in text[1]


def test_cli_csv_tokens(monkeypatch, tmp_path):
    def fake_analyze(
        _bytes,
        layoutlm_model_dir=None,
        lang="eng+deu",
        layoutlm_lang=None,
        regex_rules_path=None,
        regex_debug=False,
        layoutlm_token_model_dir=None,
    ):
        class Dummy:
            def to_dict(self, schema: bool = True):
                return {
                    "ocr": {
                        "doc_type": "unknown",
                        "language": "eng+deu",
                        "fields": {},
                        "pages": [
                            {
                                "page": 1,
                                "tokens": [
                                    {"text": "Hello", "bbox": [0, 0, 1, 1], "label": "B-TEST"},
                                ],
                            }
                        ],
                    }
                }

        return Dummy()

    sample = tmp_path / "doc.pdf"
    sample.write_bytes(b"%PDF-1.4")
    csv_path = tmp_path / "tokens.csv"

    monkeypatch.setattr(cli, "analyze_document_bytes", fake_analyze)
    monkeypatch.setattr("sys.argv", ["caesar-ocr", "analyze", str(sample), "--csv-tokens", str(csv_path)])

    cli.main()
    text = csv_path.read_text().splitlines()
    assert "text" in text[0]
    assert "Hello" in text[1]


def test_cli_csv_token_labels(monkeypatch, tmp_path):
    def fake_analyze(
        _bytes,
        layoutlm_model_dir=None,
        lang="eng+deu",
        layoutlm_lang=None,
        regex_rules_path=None,
        regex_debug=False,
        layoutlm_token_model_dir=None,
    ):
        class Dummy:
            def to_dict(self, schema: bool = True):
                return {
                    "ocr": {
                        "doc_type": "unknown",
                        "language": "eng+deu",
                        "fields": {},
                        "pages": [
                            {
                                "page": 1,
                                "tokens": [
                                    {"text": "Hello", "label": "B-TEST"},
                                    {"text": "World", "label": "I-TEST"},
                                    {"text": "Again", "label": "I-TEST"},
                                ],
                            }
                        ],
                    }
                }

        return Dummy()

    sample = tmp_path / "doc.pdf"
    sample.write_bytes(b"%PDF-1.4")
    csv_path = tmp_path / "labels.csv"

    monkeypatch.setattr(cli, "analyze_document_bytes", fake_analyze)
    monkeypatch.setattr("sys.argv", ["caesar-ocr", "analyze", str(sample), "--csv-token-labels", str(csv_path)])

    cli.main()
    text = csv_path.read_text().splitlines()
    assert "label_counts" in text[0]
    assert "B-TEST" in text[1]
