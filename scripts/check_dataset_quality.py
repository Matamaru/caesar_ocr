"""Run data quality checks for LayoutLM JSONL datasets."""

from __future__ import annotations

import argparse
import json
import pathlib
from typing import Dict, List

from caesar_ocr.layoutlm.datasets import iter_jsonl, quality_checks, validate_record


def main() -> None:
    parser = argparse.ArgumentParser(description="Check JSONL dataset quality.")
    parser.add_argument("--input", type=pathlib.Path, required=True)
    parser.add_argument("--output", type=pathlib.Path, default=None)
    args = parser.parse_args()

    results: List[Dict[str, object]] = []
    for rec in iter_jsonl(args.input):
        errors = validate_record(rec)
        qc = quality_checks(rec)
        qc["id"] = rec.id
        qc["doc_id"] = rec.doc_id
        qc["page"] = rec.page
        qc["errors"] = errors
        results.append(qc)

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(results, ensure_ascii=False, indent=2))
    else:
        print(json.dumps(results, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
