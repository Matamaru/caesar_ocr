"""Build JSONL training data from images using OCR tokens.

Outputs:
- tasks.json (for labeling tools)
- layoutlm.jsonl (token-level records with placeholder labels)
- layoutlm_train.jsonl / layoutlm_eval.jsonl (split)
"""

from __future__ import annotations

import argparse
import json
import pathlib
import random
from typing import Dict, Iterable, List, Set, Tuple

from PIL import Image
import pytesseract


def _extract_tokens(image_path: pathlib.Path, *, lang: str, psm: int) -> Tuple[str, List[Dict[str, object]]]:
    image = Image.open(image_path).convert("RGB")
    config = f"--psm {psm}"
    data = pytesseract.image_to_data(image, lang=lang, config=config, output_type=pytesseract.Output.DICT)
    tokens: List[Dict[str, object]] = []
    n = len(data["text"])
    for i in range(n):
        text = (data["text"][i] or "").strip()
        if not text:
            continue
        left = int(data["left"][i])
        top = int(data["top"][i])
        width = int(data["width"][i])
        height = int(data["height"][i])
        tokens.append(
            {
                "text": text,
                "bbox": [left, top, left + width, top + height],
                "block_num": int(data["block_num"][i]),
                "par_num": int(data["par_num"][i]),
                "line_num": int(data["line_num"][i]),
                "word_num": int(data["word_num"][i]),
            }
        )

    tokens.sort(key=lambda t: (t["block_num"], t["par_num"], t["line_num"], t["word_num"]))

    text_parts = []
    cursor = 0
    for token in tokens:
        if text_parts:
            text_parts.append(" ")
            cursor += 1
        start = cursor
        text_parts.append(token["text"])
        cursor += len(token["text"])
        end = cursor
        token["start"] = start
        token["end"] = end

    full_text = "".join(text_parts)
    return full_text, tokens


def _iter_batches(items: List[pathlib.Path], batch_size: int) -> Iterable[List[pathlib.Path]]:
    for i in range(0, len(items), batch_size):
        yield items[i : i + batch_size]


def _load_processed_images(path: pathlib.Path) -> Set[str]:
    if not path.exists():
        return set()
    processed = set()
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        try:
            rec = json.loads(line)
        except json.JSONDecodeError:
            continue
        image = rec.get("image")
        if image:
            processed.add(image)
    return processed


def _task_key(doc_id: str, page: int, image: str) -> str:
    return f"{doc_id}::{page}::{image}"


def _write_tasks(path: pathlib.Path, tasks: List[Dict[str, object]], fmt: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fmt == "json":
        path.write_text(json.dumps(tasks, ensure_ascii=False, indent=2))
        return
    with path.open("w", encoding="utf-8") as f:
        for task in tasks:
            f.write(json.dumps(task, ensure_ascii=False) + "\n")


def _collect_images(input_dir: pathlib.Path, exts: List[str]) -> List[pathlib.Path]:
    images: List[pathlib.Path] = []
    for ext in exts:
        images.extend(input_dir.glob(f"*.{ext}"))
    return sorted(set(images))


def main() -> None:
    parser = argparse.ArgumentParser(description="Build LayoutLM JSONL data from images using OCR tokens.")
    parser.add_argument("--input-dir", type=pathlib.Path, required=True, help="Directory with images")
    parser.add_argument("--output-dir", type=pathlib.Path, required=True, help="Output directory")
    parser.add_argument("--eval-ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--batch-size", type=int, default=100)
    parser.add_argument("--resume", action="store_true", help="Append to existing outputs and skip processed images")
    parser.add_argument("--lang", default="eng+deu", help="Tesseract language(s), e.g. eng, deu, eng+deu")
    parser.add_argument("--psm", type=int, default=6, help="Tesseract page segmentation mode")
    parser.add_argument("--write-tasks", action="store_true", help="Write Label Studio-style tasks")
    parser.add_argument("--tasks-format", choices=["json", "jsonl"], default="jsonl")
    parser.add_argument(
        "--ext",
        action="append",
        default=["png", "jpg", "jpeg", "tif", "tiff"],
        help="Image extension to include (repeatable)",
    )
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    tasks_path = args.output_dir / ("tasks.json" if args.tasks_format == "json" else "tasks.jsonl")
    all_path = args.output_dir / "layoutlm.jsonl"
    train_path = args.output_dir / "layoutlm_train.jsonl"
    eval_path = args.output_dir / "layoutlm_eval.jsonl"

    if not args.resume:
        for path in (tasks_path, all_path, train_path, eval_path):
            if path.exists():
                path.unlink()

    images = _collect_images(args.input_dir, args.ext)
    if not images:
        raise SystemExit("No images found.")

    rng = random.Random(args.seed)
    total = 0
    train_count = 0
    eval_count = 0
    processed = _load_processed_images(all_path) if args.resume else set()
    total_images = len(images) - len(processed)
    processed_images = 0

    tasks: List[Dict[str, object]] = []
    with all_path.open("a", encoding="utf-8") as all_f, \
         train_path.open("a", encoding="utf-8") as train_f, \
         eval_path.open("a", encoding="utf-8") as eval_f:
        for batch in _iter_batches(images, args.batch_size):
            for image_path in batch:
                if args.resume and str(image_path) in processed:
                    continue
                full_text, tokens = _extract_tokens(image_path, lang=args.lang, psm=args.psm)
                with Image.open(image_path) as im:
                    width, height = im.size

                doc_id = image_path.stem
                page_idx = 1

                task = {
                    "data": {
                        "text": full_text,
                        "image": str(image_path),
                        "doc_id": doc_id,
                        "page": page_idx,
                        "task_id": _task_key(doc_id, page_idx, str(image_path)),
                    },
                    "meta": {
                        "tokens": tokens,
                        "image_size": {"width": width, "height": height},
                    },
                }
                if args.write_tasks:
                    tasks.append(task)

                record = {
                    "id": None,
                    "image": str(image_path),
                    "text": full_text,
                    "doc_id": doc_id,
                    "page": page_idx,
                    "tokens": [t["text"] for t in tokens],
                    "bboxes": [t["bbox"] for t in tokens],
                    "labels": ["O"] * len(tokens),
                    "spans": [],
                }
                all_f.write(json.dumps(record, ensure_ascii=False) + "\n")

                if rng.random() < args.eval_ratio:
                    eval_f.write(json.dumps(record, ensure_ascii=False) + "\n")
                    eval_count += 1
                else:
                    train_f.write(json.dumps(record, ensure_ascii=False) + "\n")
                    train_count += 1
                total += 1
                processed_images += 1

            if total_images > 0:
                pct = (processed_images / total_images) * 100
                print(f"Progress: {processed_images}/{total_images} ({pct:.1f}%)")
            else:
                print(f"Processed batch of {len(batch)} images (total {total})")

    if args.write_tasks:
        _write_tasks(tasks_path, tasks, args.tasks_format)
        print(f"Wrote {len(tasks)} tasks to {tasks_path}")
    print(f"Wrote {total} records to {all_path}")
    print(f"Wrote {train_count} train records to {train_path}")
    print(f"Wrote {eval_count} eval records to {eval_path}")


if __name__ == "__main__":
    main()
