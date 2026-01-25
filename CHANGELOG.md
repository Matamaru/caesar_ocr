# Changelog

## v0.3.1.dev0 - 2026-01-25
- Development version bump after schema, regex, IO wiring, and CLI refactors.

## v0.2.0 - 2026-01-25 Phase 0 Baseline and Structure
- Restructured package into clean OCR + LayoutLMv3 layout.
- Added OCR engine split (engine + tesseract adapter + postprocess utilities).
- Added dataset schema + validation scaffolding and canonical output schemas.
- Added CLI entry points for analyze/train/infer and LayoutLM inference helper.
- Added configurable OCR/Language flags and defaults.
- Added dataset prep scripts (render PNGs, build JSONL with task support).
- Removed legacy apps/graph and unrelated hub database tooling.
