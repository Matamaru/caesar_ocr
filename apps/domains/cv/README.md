# CV Domain Pack

This domain pack targets CV classification and data extraction.

Scope
- CV type classification (e.g., Lebenslauf, Europass, modern résumé, academic CV)
- Block segmentation (personal info, work history, education, skills)
- Field extraction and normalization

CV Types
- See `apps/domains/cv/types.md` for the initial taxonomy and layout characteristics.

Folders
- `sample_docs/`: synthetic or real CV samples (ignored by git)
- `expected/`: expected outputs for regression tests

Notes
- This pack is separate from the core OCR/LayoutLM pipeline.
