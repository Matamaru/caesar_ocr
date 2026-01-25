# Caesar OCR Roadmap

This roadmap resets the project with a clear architecture for OCR + LayoutLMv3, a hybrid regex system (YAML rules + Python plugins), and a JSONL training pipeline. Dates are not assigned yet; tasks are ordered by dependency.

## Phase 0 — Baseline and Structure (Completed)
- Define canonical output schema (JSON) and dataset schema (JSONL). ✅
- Establish package layout: `io/`, `ocr/`, `layoutlm/`, `regex/`, `pipeline/`. ✅
- Add minimal CLI scaffolding for: `analyze`, `train`, `infer`. ✅
- Decide language flags (`en`, `de`, `en+de`) and propagate through OCR + LayoutLM. ✅
- Introduce configuration defaults (DPI, PSM, language) in `config/defaults.yaml`. ✅

## Phase 1 — OCR Core (Completed)
- Implement PDF + image ingestion with page metadata. ✅
- Add OCR engine wrapper (Tesseract initially) with pre/post-processing hooks. ✅
- Split OCR into engine + tesseract adapter + postprocess utilities. ✅
- Normalize text + tokens, keep positional boxes. ✅
- Unit tests for OCR I/O and normalization. ✅
- Move image loading + PDF page handling into `io/loaders.py` and wire it into the OCR engine. ✅

## Phase 2 — Hybrid Regex System (Completed)
- YAML rule format (pattern, group, output_field, confidence, validators). ✅
- Python plugin interface for advanced extraction (multi-line, conditional). ✅
- Rule runner with deterministic order and debug traces. ✅
- Tests for YAML parsing and plugin execution. ✅
- Implement YAML rule runner and integrate it into the pipeline. ✅

## Phase 3 — LayoutLMv3 Integration (Completed)
- Document classifier and token classifier adapters. ✅
- Unified inference pipeline: OCR -> tokens/boxes -> LayoutLMv3 -> merge. ✅
- Language flag support for `en`, `de`, `en+de` in both OCR + LayoutLM. ✅
- Minimal evaluation metrics (precision/recall/F1 per label). ✅
- Include token-classifier outputs in schema per page. ✅
- Add unified CLI mode to run doc + token classifiers together. ✅

## Phase 4 — Training Pipeline (JSONL) (Completed)
- JSONL dataset schema with validation. ✅
- Train/eval split tooling and data quality checks. ✅
- Training script for token classification (LayoutLMv3). ✅
- Inference script for JSONL and raw documents. ✅

## Phase 5 — Output Formats (Completed)
- Canonical JSON output from pipeline. ✅
- CSV export for fields/entities. ✅

## Phase 6 — Apps/Domain Workflows (Planned)
Goal: provide learning-friendly example datasets and rule packs without affecting core OCR/LayoutLM.

- Domain rule packs (YAML + plugins) for invoices, certificates, CVs, passports.
- Sample documents and synthetic generators for each domain.
- Example JSONL datasets derived from sample docs.
- Regression tests per domain pack with expected outputs.
- Minimal CLI helpers to generate synthetic docs for user-specific tests.
- Documentation and templates so users can add their own domain packs.

Proposed structure (separate from core):

```
apps/
  domains/
    invoices/
      rules.yaml
      plugins.py
      sample_docs/
      expected/
      generate.py
      README.md
    certificates/
      rules.yaml
      plugins.py
      sample_docs/
      expected/
      generate.py
      README.md
    cv/
      rules.yaml
      plugins.py
      sample_docs/
      expected/
      generate.py
      README.md
    passports/
      rules.yaml
      plugins.py
      sample_docs/
      expected/
      generate.py
      README.md
scripts/
  build_domain_jsonl.py
  run_domain_regression.py
```

## Phase 7 — Hardening & Ops
- Config-driven defaults and environment overrides.
- Performance profiling and caching.
- Logging, tracing, and error reporting.
- Packaging and versioning.
