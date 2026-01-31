"""Generate passport PDFs and a JSONL manifest with ground truth."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from apps.domains.passport.generate import generate_passports


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate passport eval set + manifest.")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--count", type=int, default=20)
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()

    generate_passports(
        args.output_dir,
        count=args.count,
        seed=args.seed,
        manifest_path=args.manifest,
    )
    print(f"Wrote manifest to {args.manifest}")


if __name__ == "__main__":
    main()
