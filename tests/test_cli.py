import json
from pathlib import Path

import caesar_ocr.cli as cli


def test_cli_outputs_json(monkeypatch, tmp_path, capsys):
    def fake_analyze(_bytes, layoutlm_model_dir=None, lang="eng+deu", layoutlm_lang=None, regex_rules_path=None, regex_debug=False):
        class Dummy:
            def to_dict(self, schema: bool = True):
                return {"ok": True, "lang": lang}

        return Dummy()

    sample = tmp_path / "doc.pdf"
    sample.write_bytes(b"%PDF-1.4")

    monkeypatch.setattr(cli, "analyze_document_bytes", fake_analyze)
    monkeypatch.setattr("sys.argv", ["caesar-ocr", str(sample)])

    cli.main()
    out = capsys.readouterr().out.strip()
    data = json.loads(out)
    assert data["ok"] is True
    assert data["lang"] == "eng+deu"


def test_cli_writes_output_file(monkeypatch, tmp_path):
    def fake_analyze(_bytes, layoutlm_model_dir=None, lang="eng+deu", layoutlm_lang=None, regex_rules_path=None, regex_debug=False):
        class Dummy:
            def to_dict(self, schema: bool = True):
                return {"ok": True}

        return Dummy()

    sample = tmp_path / "doc.pdf"
    sample.write_bytes(b"%PDF-1.4")
    out_path = tmp_path / "out.json"

    monkeypatch.setattr(cli, "analyze_document_bytes", fake_analyze)
    monkeypatch.setattr("sys.argv", ["caesar-ocr", str(sample), "--output", str(out_path)])

    cli.main()
    assert json.loads(out_path.read_text()) == {"ok": True}
