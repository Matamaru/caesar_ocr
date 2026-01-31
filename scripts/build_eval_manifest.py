"""Build a JSONL manifest for eval_fields.py from a folder or glob."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable, List


def _iter_paths(path_str: str) -> Iterable[Path]:
    path = Path(path_str)
    if path.is_dir():
        for ext in ("*.pdf", "*.png", "*.jpg", "*.jpeg"):
            for item in sorted(path.glob(ext)):
                yield item
        return
    for item in sorted(Path().glob(path_str)):
        if item.is_file():
            yield item


def main() -> None:
    parser = argparse.ArgumentParser(description="Create a JSONL eval manifest from files.")
    parser.add_argument("input", help="Directory or glob pattern of documents.")
    parser.add_argument("output", type=Path, help="Output JSONL path.")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of files.")
    parser.add_argument("--doc-type", default=None, help="Optional doc_type label to include.")
    args = parser.parse_args()

    items: List[Path] = list(_iter_paths(args.input))
    if args.limit is not None:
        items = items[: args.limit]

    rows = []
    for item in items:
        row = {"path": str(item), "expected": {}}
        if args.doc_type:
            row["doc_type"] = args.doc_type
        rows.append(row)

    args.output.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")
    print(f"Wrote {len(rows)} entries to {args.output}")


if __name__ == "__main__":
    main()
