import yaml
from pathlib import Path


def test_defaults_yaml_has_expected_keys():
    path = Path(__file__).resolve().parents[1] / "caesar_ocr" / "config" / "defaults.yaml"
    data = yaml.safe_load(path.read_text())

    assert "ocr" in data
    assert "layoutlm" in data
    assert data["ocr"]["dpi"] == 300
    assert data["ocr"]["psm"] == 6
    assert data["ocr"]["lang"] == "eng+deu"
    assert data["layoutlm"]["device"] == "auto"
