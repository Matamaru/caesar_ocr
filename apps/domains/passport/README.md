# Passport Domain Pack

Purpose
- Extract MRZ (machine readable zone) and core passport fields.
- Provide lightweight heuristics for document classification.

Files
- `rules.yaml`: regex rules for MRZ lines
- `plugins.py`: MRZ parsing helpers
- `sample_docs/`: optional PDFs (ignored by git)
- `expected/`: expected outputs for regression tests
