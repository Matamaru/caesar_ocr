import io
import json
from pathlib import Path

from PIL import Image

from caesar_ocr.io.loaders import load_images_from_bytes
from caesar_ocr.io.writers import write_csv, write_json, write_jsonl


def _png_bytes(size=(10, 10), color=(255, 255, 255)) -> bytes:
    image = Image.new("RGB", size, color)
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    return buf.getvalue()


def test_load_images_from_bytes_image():
    pages = load_images_from_bytes(_png_bytes())
    assert len(pages) == 1
    assert pages[0].page == 1
    assert pages[0].width == 10
    assert pages[0].height == 10


def test_write_json(tmp_path: Path):
    path = tmp_path / "out.json"
    data = {"a": 1}
    write_json(path, data)
    assert json.loads(path.read_text()) == data


def test_write_jsonl(tmp_path: Path):
    path = tmp_path / "out.jsonl"
    rows = [{"a": 1}, {"b": 2}]
    write_jsonl(path, rows)
    lines = [json.loads(line) for line in path.read_text().splitlines()]
    assert lines == rows


def test_write_csv(tmp_path: Path):
    path = tmp_path / "out.csv"
    rows = [{"a": "1", "b": "2"}, {"a": "3", "b": "4"}]
    write_csv(path, rows, fieldnames=["a", "b"])
    text = path.read_text().splitlines()
    assert text[0] == "a,b"
    assert text[1] == "1,2"
    assert text[2] == "3,4"
