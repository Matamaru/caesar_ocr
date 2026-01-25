from __future__ import annotations

import json
from pathlib import Path

from PIL import Image

import importlib.util
import sys


def _load_build_jsonl():
    script_path = Path(__file__).resolve().parents[1] / "scripts" / "build_jsonl.py"
    spec = importlib.util.spec_from_file_location("build_jsonl", script_path)
    if spec is None or spec.loader is None:
        raise RuntimeError("Failed to load build_jsonl script")
    module = importlib.util.module_from_spec(spec)
    sys.modules["build_jsonl"] = module
    spec.loader.exec_module(module)
    return module


build_jsonl = _load_build_jsonl()


def test_build_jsonl_with_pdf(tmp_path: Path, monkeypatch):
    input_dir = tmp_path / "input"
    output_dir = tmp_path / "output"
    input_dir.mkdir()
    output_dir.mkdir()

    pdf_path = input_dir / "report.pdf"
    pdf_path.write_bytes(b"%PDF-1.4\n%dummy")

    dummy_image = Image.new("RGB", (100, 50), color=(255, 255, 255))

    class DummyPage:
        def __init__(self, page: int):
            self.page = page
            self.image = dummy_image
            self.width = dummy_image.width
            self.height = dummy_image.height

    monkeypatch.setattr(
        build_jsonl, "load_images_from_bytes", lambda _b, dpi=300: [DummyPage(1), DummyPage(2)]
    )

    def fake_ocr_tokens(_image, *, lang, psm):
        tokens = [
            {"text": "Hello", "bbox": [0, 0, 10, 10]},
            {"text": "World", "bbox": [12, 0, 30, 10]},
        ]
        return "Hello World", tokens

    monkeypatch.setattr(build_jsonl, "ocr_tokens_from_image", fake_ocr_tokens)

    monkeypatch.setattr(
        "sys.argv",
        [
            "build_jsonl.py",
            "--input-dir",
            str(input_dir),
            "--output-dir",
            str(output_dir),
            "--eval-ratio",
            "0.0",
        ],
    )

    build_jsonl.main()

    jsonl_path = output_dir / "layoutlm.jsonl"
    assert jsonl_path.exists()
    records = [json.loads(line) for line in jsonl_path.read_text().splitlines() if line.strip()]
    assert len(records) == 2
    assert records[0]["page"] == 1
    assert records[1]["page"] == 2
    assert records[0]["source_pdf"] == str(pdf_path)
    assert records[0]["tokens"] == ["Hello", "World"]
    assert records[0]["bboxes"] == [[0, 0, 10, 10], [12, 0, 30, 10]]

    images_dir = output_dir / "images"
    assert (images_dir / "report_page_001.png").exists()
    assert (images_dir / "report_page_002.png").exists()
