# caesar_ocr

Pure OCR + LayoutLMv3 package for document understanding. Includes a simple CLI,
Python APIs, and scripts for dataset preparation and training workflows.

## Install

```bash
pip install -e .
```

## CLI

```bash
caesar-ocr /path/to/file.pdf --output result.json
```

Train and infer helpers:

```bash
caesar-ocr-train --train layoutlm_train.jsonl --eval layoutlm_eval.jsonl --output-dir models/layoutlmv3-token
caesar-ocr-infer /path/to/file.pdf --model-dir models/layoutlmv3-token --output result.json
caesar-ocr-infer-token --model-dir models/layoutlmv3-token --input sample.jsonl --output predictions.jsonl
```

## Python usage

```python
from pathlib import Path

from caesar_ocr import analyze_bytes, analyze_bytes_layoutlm, analyze_document_bytes

payload = Path("document.pdf").read_bytes()
result = analyze_bytes(payload)

print(result.doc_type)
print(result.fields)
print(result.ocr_text)

layoutlm_result = analyze_bytes_layoutlm(payload, model_dir="path/to/layoutlmv3-model")
print(layoutlm_result.doc_type)
print(layoutlm_result.scores)

tool_result = analyze_document_bytes(
    payload,
    layoutlm_model_dir="path/to/layoutlmv3-model",
)
print(tool_result.to_dict())
```

## Scripts (root `scripts/`)

### `scripts/split_jsonl.py`
Split a JSONL dataset into train/val.

```bash
python scripts/split_jsonl.py \
  --input layoutlm.jsonl \
  --train train.jsonl \
  --val val.jsonl
```

### `scripts/train_layoutlmv3_token.py`
Train a LayoutLMv3 token classifier from JSONL.

```bash
python scripts/train_layoutlmv3_token.py \
  --train layoutlm_train.jsonl \
  --eval layoutlm_eval.jsonl \
  --output-dir models/layoutlmv3-token
```

### `scripts/infer_layoutlmv3_token.py`
Run LayoutLMv3 token classification on JSONL or image/PDF input.

```bash
python scripts/infer_layoutlmv3_token.py \
  --model-dir models/layoutlmv3-token \
  --input sample.jsonl \
  --output predictions.jsonl
```

### `scripts/train_layoutlmv3.py`
Train a LayoutLMv3 document classifier with fixed paths/labels.

```bash
python scripts/train_layoutlmv3.py
```

### `scripts/build_jsonl.py`
Build JSONL datasets from images using OCR tokens (train/eval split included).

### `scripts/render_pngs.py`
Render PDF pages to PNGs for dataset preparation.

## Package layout

- `caesar_ocr/ocr/` OCR engine + post-processing
- `caesar_ocr/layoutlm/` LayoutLMv3 inference + training utilities
- `caesar_ocr/pipeline/` end-to-end analysis
- `caesar_ocr/regex/` hybrid rule system (YAML + Python plugins)
- `caesar_ocr/io/` loaders and output writers
