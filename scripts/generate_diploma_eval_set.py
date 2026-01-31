"""Generate diploma PDFs and a JSONL manifest with ground truth."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from apps.domains.diploma.generate import generate_diplomas


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate diploma eval set + manifest.")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--count", type=int, default=20)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--lang", choices=["de", "en", "both"], default="de")
    args = parser.parse_args()

    generate_diplomas(
        args.output_dir,
        count=args.count,
        seed=args.seed,
        lang=args.lang,
        manifest_path=args.manifest,
    )
    print(f"Wrote manifest to {args.manifest}")


if __name__ == "__main__":
    main()
