import json
from pathlib import Path

from caesar_ocr.layoutlm.datasets import LayoutLMTokenRecord, iter_jsonl, validate_record
from caesar_ocr.layoutlm.metrics import precision_recall_f1


def test_iter_jsonl_and_validate(tmp_path: Path):
    path = tmp_path / "data.jsonl"
    record = {
        "id": "1",
        "image": "img.png",
        "text": "hello world",
        "doc_id": "doc",
        "page": 1,
        "tokens": ["hello", "world"],
        "bboxes": [[0, 0, 10, 10], [20, 0, 30, 10]],
        "labels": ["O", "B-TEST"],
        "spans": [],
    }
    path.write_text(json.dumps(record) + "\n")

    records = list(iter_jsonl(path))
    assert len(records) == 1
    rec = records[0]
    assert isinstance(rec, LayoutLMTokenRecord)
    assert rec.tokens == ["hello", "world"]

    errors = validate_record(rec)
    assert errors == []


def test_validate_record_errors():
    rec = LayoutLMTokenRecord(
        id=None,
        image=None,
        text="",
        doc_id=None,
        page=None,
        tokens=["a"],
        bboxes=[],
        labels=["O"],
        spans=[],
    )
    errors = validate_record(rec)
    assert "text is empty" in errors
    assert "tokens and bboxes length mismatch" in errors


def test_precision_recall_f1():
    y_true = ["A", "A", "B", "O"]
    y_pred = ["A", "B", "B", "O"]
    metrics = precision_recall_f1(y_true, y_pred, labels=["A", "B", "O"])

    assert metrics["A"]["precision"] == 1.0
    assert metrics["A"]["recall"] == 0.5
    assert metrics["B"]["precision"] == 0.5
    assert metrics["B"]["recall"] == 1.0
    assert metrics["O"]["precision"] == 1.0
    assert metrics["O"]["recall"] == 1.0


def test_train_token_compute_metrics_flattened():
    from caesar_ocr.layoutlm.metrics import precision_recall_f1

    y_true = ["A", "A", "B", "O"]
    y_pred = ["A", "B", "B", "O"]
    per_label = precision_recall_f1(y_true, y_pred, labels=["A", "B", "O"])
    flat = {}
    for label, metrics in per_label.items():
        flat[f"{label}_precision"] = float(metrics["precision"])
        flat[f"{label}_recall"] = float(metrics["recall"])
        flat[f"{label}_f1"] = float(metrics["f1"])
        flat[f"{label}_support"] = float(metrics["support"])

    assert flat["A_precision"] == 1.0
    assert flat["A_recall"] == 0.5
    assert flat["B_precision"] == 0.5
    assert flat["B_recall"] == 1.0
