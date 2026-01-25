"""CLI to run LayoutLMv3 inference on a PDF/image."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .layoutlm.infer import analyze_bytes_layoutlm


def main() -> None:
    parser = argparse.ArgumentParser(description="Run LayoutLMv3 inference on a document.")
    parser.add_argument("path", help="Path to PDF or image")
    parser.add_argument("--model-dir", required=True, help="LayoutLMv3 model directory")
    parser.add_argument("--output", type=Path, default=None, help="Write JSON output to file")
    parser.add_argument("--labels", nargs="*", default=None, help="Optional label list")
    parser.add_argument("--lang", default=None, help="LayoutLM language flag (e.g. en, de)")
    args = parser.parse_args()

    payload = Path(args.path).read_bytes()
    result = analyze_bytes_layoutlm(payload, model_dir=args.model_dir, labels=args.labels, lang=args.lang)
    data = {
        "doc_type": result.doc_type,
        "label_id": result.label_id,
        "scores": result.scores,
    }

    if args.output:
        args.output.write_text(json.dumps(data, ensure_ascii=False, indent=2))
    else:
        print(json.dumps(data, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
