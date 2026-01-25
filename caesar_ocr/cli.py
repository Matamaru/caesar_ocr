"""Simple CLI for OCR and optional LayoutLM classification."""

import argparse
import json
from pathlib import Path

from .pipeline.analyze import analyze_document_bytes


def main() -> None:
    parser = argparse.ArgumentParser(description="Run OCR and optional LayoutLM on a document.")
    parser.add_argument("path", nargs="?", help="Path to PDF or image")
    parser.add_argument("--output", type=Path, default=None, help="Write JSON output to file")
    parser.add_argument("--layoutlm-model-dir", default=None, help="LayoutLM model directory")
    parser.add_argument("--lang", default="eng+deu", help="OCR language(s), e.g. eng, deu, eng+deu")
    parser.add_argument(
        "--layoutlm-lang",
        default=None,
        help="LayoutLM language flag (e.g. en, de). If omitted, processor default is used.",
    )
    parser.add_argument("--regex-rules", default=None, help="Path to YAML regex rules")
    parser.add_argument("--regex-debug", action="store_true", help="Include regex debug trace in output")
    parser.add_argument("--pick", action="store_true", help="Not supported in this minimal CLI")
    args = parser.parse_args()

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
    )
    data = result.to_dict()

    if args.output:
        args.output.write_text(json.dumps(data, ensure_ascii=False, indent=2))
    else:
        print(json.dumps(data, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
