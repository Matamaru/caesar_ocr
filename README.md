# caesar_ocr

![Python](https://img.shields.io/badge/python-3.x-blue)
![Status](https://img.shields.io/badge/status-alpha-orange)
![License](https://img.shields.io/badge/license-MIT-green)

OCR + LayoutLMv3 package for document understanding. Includes a CLI, Python APIs,
JSONL dataset tooling, and a hybrid regex system (YAML rules + Python plugins).

## Install

```bash
pip install -e .
```

## CLI

Analyze a PDF/image:

```bash
caesar-ocr analyze /path/to/file.pdf --output result.json
```

With regex rules and LayoutLM (doc + token):

```bash
caesar-ocr analyze /path/to/file.pdf \
  --output result.json \
  --regex-rules rules.yaml \
  --regex-debug \
  --layoutlm-model-dir models/layoutlmv3-doc \
  --layoutlm-token-model-dir models/layoutlmv3-token \
  --layoutlm-lang en
```

CSV export of extracted fields:

```bash
caesar-ocr analyze /path/to/file.pdf --csv-fields fields.csv
```

Token CSV export (one row per token):

```bash
caesar-ocr analyze /path/to/file.pdf --csv-tokens tokens.csv
```

Token label counts per page:

```bash
caesar-ocr analyze /path/to/file.pdf --csv-token-labels token_labels.csv
```

Document type hints (keyword-based) are added automatically as `doc_hints`
in the JSON output.

Train and infer helpers:

```bash
caesar-ocr-train --train layoutlm_train.jsonl --eval layoutlm_eval.jsonl --output-dir models/layoutlmv3-token
caesar-ocr-infer /path/to/file.pdf --model-dir models/layoutlmv3-doc --output result.json
caesar-ocr-infer-token --model-dir models/layoutlmv3-token --input sample.jsonl --output predictions.jsonl
```

## Python usage

```python
from pathlib import Path

from caesar_ocr import analyze_bytes, analyze_bytes_layoutlm, analyze_document_bytes
from caesar_ocr.domain_samples import (
    generate_cv_samples,
    generate_diploma_samples,
    generate_fehlerprotokoll_samples,
    generate_passport_samples,
)

payload = Path("document.pdf").read_bytes()
result = analyze_bytes(payload, lang="eng+deu")

print(result.doc_type)
print(result.fields)
print(result.ocr_text)

layoutlm_result = analyze_bytes_layoutlm(payload, model_dir="path/to/layoutlmv3-model", lang="en")
print(layoutlm_result.doc_type)
print(layoutlm_result.scores)

tool_result = analyze_document_bytes(
    payload,
    layoutlm_model_dir="path/to/layoutlmv3-model",
    lang="eng+deu",
    layoutlm_lang="en",
    regex_rules_path="rules.yaml",
    regex_debug=True,
)
print(tool_result.to_dict())

# Generate sample domain PDFs
generate_passport_samples("samples/passports", count=10)
generate_diploma_samples("samples/diplomas", count=10, lang="both")
generate_cv_samples("samples/cv", count=10, all_types=True)
generate_fehlerprotokoll_samples("samples/fehlerprotokoll", report_date="2025-11-30")
```

## Hybrid regex rules

Rules live in YAML and run in file order. Each rule can either define a regex
or a plugin. Validators are optional callbacks registered in code.

Example:

```yaml
- name: invoice_number
  pattern: '(?i)invoice\s*(no|number)?\s*[:#-]?\s*([A-Z0-9-]{3,})'
  group: 2
  output_field: invoice_number
  confidence: 0.85
  validators: [is_invoice]

- name: custom_plugin
  plugin: my_plugin
```

Supported fields:
- `name`: rule identifier
- `pattern`: regex pattern (if omitted, rule is skipped unless `plugin` is set)
- `group`: capture group to extract (default `0`)
- `output_field`: output field name (defaults to `name`)
- `confidence`: numeric confidence (optional) stored as `<field>_confidence`
- `flags`: regex flags string (e.g., `IM`, `S`)
- `plugin`: plugin name (function callback)
- `validators`: list of validator names (function callbacks)

When `--regex-debug` is enabled, output includes `__debug__` entries per rule.

## Scripts (root `scripts/`)

### `scripts/build_jsonl.py`
Build JSONL datasets from images or PDFs using OCR tokens (train/eval split included).
Supports PDF page extraction, progress bar, and file filtering.

Example:
```bash
python scripts/build_jsonl.py \
  --input-dir apps/domains/fehlerprotokoll/sample_docs \
  --output-dir apps/domains/fehlerprotokoll/labels \
  --lang deu \
  --write-tasks --tasks-format jsonl \
  --eval-ratio 0.1 \
  --batch-size 3 \
  --progress-bar
```

### `scripts/render_pngs.py`
Render PDF pages to PNGs for dataset preparation.

### `scripts/split_jsonl.py`
Split a JSONL dataset into train/val.

### `scripts/check_dataset_quality.py`
Run data quality checks on JSONL datasets (bbox bounds, label coverage, mismatches).

## Domain packs (apps/)

Domain packs live under `apps/domains/` and are separate from the core OCR/LayoutLM code.

Included packs:
- `apps/domains/fehlerprotokoll/` (error reports, generator + regex + training data)
- `apps/domains/cv/` (CV types + generator + rules)
- `apps/domains/passport/` (MRZ rules + parser)
- `apps/domains/diploma/` (degree rules + extractor)

## Package layout

- `caesar_ocr/ocr/` OCR engine + post-processing (tokens + bboxes)
- `caesar_ocr/layoutlm/` LayoutLMv3 inference + training utilities
- `caesar_ocr/pipeline/` end-to-end analysis + schema output
- `caesar_ocr/regex/` hybrid rule system (YAML + Python plugins)
- `caesar_ocr/io/` loaders and output writers

## Models (local, gitignored)

Place domain models under `models/<domain>/` (gitignored). Example:

```
models/
  invoices/
    layoutlmv3-doc/
    layoutlmv3-token/
```

## License

MIT License. See `LICENSE`.

## Auto-label helpers

- `scripts/auto_label_fehlerprotokoll.py` (Fehlerprotokoll bootstrap labels)
- `scripts/auto_label_cv.py` (CV bootstrap labels)
