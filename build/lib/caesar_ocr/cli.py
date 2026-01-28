"""Simple CLI for OCR and optional LayoutLM classification."""

import argparse
import json
import sys
from pathlib import Path

from .pipeline.analyze import analyze_document_bytes


def main() -> None:
    parser = argparse.ArgumentParser(description="Run OCR and optional LayoutLM on a document.")
    sub = parser.add_subparsers(dest="cmd")

    analyze = sub.add_parser("analyze", help="Run OCR + optional LayoutLM on a document")
    analyze.add_argument("path", nargs="?", help="Path to PDF or image")
    analyze.add_argument("--output", type=Path, default=None, help="Write JSON output to file")
    analyze.add_argument("--csv-fields", type=Path, default=None, help="Write flattened fields CSV")
    analyze.add_argument("--csv-tokens", type=Path, default=None, help="Write token-level CSV")
    analyze.add_argument("--csv-token-labels", type=Path, default=None, help="Write token label counts per page")
    analyze.add_argument("--layoutlm-model-dir", default=None, help="LayoutLM model directory")
    analyze.add_argument("--layoutlm-token-model-dir", default=None, help="LayoutLM token classifier model directory")
    analyze.add_argument("--lang", default="eng+deu", help="OCR language(s), e.g. eng, deu, eng+deu")
    analyze.add_argument(
        "--layoutlm-lang",
        default=None,
        help="LayoutLM language flag (e.g. en, de). If omitted, processor default is used.",
    )
    analyze.add_argument("--regex-rules", default=None, help="Path to YAML regex rules")
    analyze.add_argument("--regex-debug", action="store_true", help="Include regex debug trace in output")
    analyze.add_argument("--pick", action="store_true", help="Not supported in this minimal CLI")

    # Backward-compatible: allow `caesar-ocr /path/to/file.pdf`
    if len(sys.argv) > 1 and sys.argv[1] not in {"analyze"} and not sys.argv[1].startswith("-"):
        sys.argv.insert(1, "analyze")

    args = parser.parse_args()

    if args.cmd is None:
        args.cmd = "analyze"
    if args.cmd != "analyze":
        raise SystemExit(f"Unsupported command: {args.cmd}")
    if args.pick:
        raise SystemExit("--pick is not supported; provide a path instead")
    if not args.path:
        raise SystemExit("Provide a file path")

    payload = Path(args.path).read_bytes()
    result = analyze_document_bytes(
        payload,
        layoutlm_model_dir=args.layoutlm_model_dir,
        lang=args.lang,
        layoutlm_lang=args.layoutlm_lang,
        regex_rules_path=args.regex_rules,
        regex_debug=args.regex_debug,
        layoutlm_token_model_dir=args.layoutlm_token_model_dir,
    )
    data = result.to_dict()

    if args.csv_fields or args.csv_tokens or args.csv_token_labels:
        from .io.writers import flatten_fields_to_rows, token_labels_by_page_rows, tokens_to_rows, write_csv

        if args.csv_fields:
            rows = flatten_fields_to_rows(data)
            fieldnames = sorted({k for row in rows for k in row.keys()})
            write_csv(args.csv_fields, rows, fieldnames=fieldnames)
            print(f"Wrote CSV fields to {args.csv_fields}")
        if args.csv_tokens:
            rows = tokens_to_rows(data)
            fieldnames = sorted({k for row in rows for k in row.keys()}) if rows else []
            write_csv(args.csv_tokens, rows, fieldnames=fieldnames)
            print(f"Wrote CSV tokens to {args.csv_tokens}")
        if args.csv_token_labels:
            rows = token_labels_by_page_rows(data)
            fieldnames = sorted({k for row in rows for k in row.keys()}) if rows else []
            write_csv(args.csv_token_labels, rows, fieldnames=fieldnames)
            print(f"Wrote CSV token labels to {args.csv_token_labels}")

    if args.output or (not args.csv_fields and not args.csv_tokens and not args.csv_token_labels):
        payload = json.dumps(data, ensure_ascii=False, indent=2)
        if args.output:
            args.output.write_text(payload)
            print(f"Wrote JSON output to {args.output}")
        else:
            print(payload)


if __name__ == "__main__":
    main()
