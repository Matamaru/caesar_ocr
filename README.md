# caesar_ocr

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
Build JSONL datasets from images using OCR tokens (train/eval split included).

### `scripts/render_pngs.py`
Render PDF pages to PNGs for dataset preparation.

### `scripts/split_jsonl.py`
Split a JSONL dataset into train/val.

## Package layout

- `caesar_ocr/ocr/` OCR engine + post-processing (tokens + bboxes)
- `caesar_ocr/layoutlm/` LayoutLMv3 inference + training utilities
- `caesar_ocr/pipeline/` end-to-end analysis + schema output
- `caesar_ocr/regex/` hybrid rule system (YAML + Python plugins)
- `caesar_ocr/io/` loaders and output writers
