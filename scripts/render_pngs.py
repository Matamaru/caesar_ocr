"""Render PDFs to PNGs (batched)."""

from __future__ import annotations

import argparse
import pathlib
from typing import List

from pdf2image import convert_from_path


def _chunk(items: List[pathlib.Path], size: int) -> List[List[pathlib.Path]]:
    return [items[i : i + size] for i in range(0, len(items), size)]


def main() -> None:
    parser = argparse.ArgumentParser(description="Render PDFs to PNGs (batched).")
    parser.add_argument("--input-dir", type=pathlib.Path, required=True)
    parser.add_argument("--output-dir", type=pathlib.Path, required=True)
    parser.add_argument("--dpi", type=int, default=300)
    parser.add_argument("--batch-size", type=int, default=5)
    parser.add_argument("--max-pages", type=int, default=None)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    pdfs = sorted(args.input_dir.glob("*.pdf"))
    if not pdfs:
        raise SystemExit("No PDFs found.")

    for batch in _chunk(pdfs, args.batch_size):
        for pdf_path in batch:
            pages = convert_from_path(str(pdf_path), dpi=args.dpi)
            if args.max_pages is not None:
                pages = pages[: args.max_pages]
            for idx, page in enumerate(pages, start=1):
                image_name = f"{pdf_path.stem}_p{idx:03d}.png"
                image_path = args.output_dir / image_name
                page.save(image_path)
        print(f"Rendered batch of {len(batch)} PDFs")


if __name__ == "__main__":
    main()
