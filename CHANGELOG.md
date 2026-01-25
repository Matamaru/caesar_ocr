# Changelog

## v0.4.1.dev0 - 2026-01-25 Phase 2 Hybrid Regex System
- Development bump after Phase 2 completion.

## v0.4.0 - 2026-01-25 Phase 2 Hybrid Regex System
- Added validators, confidence handling, and debug traces for regex rules.
- Added default plugin/validator registries and updated docs/tests.

## v0.3.1.dev0 - 2026-01-25 Phase 1 OCR Core
- Development bump after Phase 1 completion and test coverage expansion.

## v0.3.0 - 2026-01-25 Phase 1 OCR Core
- Completed OCR core with token+bbox output and multi-page support.
- Wired canonical schema output and regex rules into the pipeline.
- Added IO loaders/writers, LayoutLM helpers, and expanded tests.
- Cleaned CLI entry points and dataset tooling.

## v0.2.0 - 2026-01-25 Phase 0 Baseline and Structure
- Restructured package into clean OCR + LayoutLMv3 layout.
- Added OCR engine split (engine + tesseract adapter + postprocess utilities).
- Added dataset schema + validation scaffolding and canonical output schemas.
- Added CLI entry points for analyze/train/infer and LayoutLM inference helper.
- Added configurable OCR/Language flags and defaults.
- Added dataset prep scripts (render PNGs, build JSONL with task support).
- Removed legacy apps/graph and unrelated hub database tooling.
