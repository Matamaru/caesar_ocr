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
import fnmatch
import sys
from typing import Dict, Iterable, List, Set, Tuple

from PIL import Image

from caesar_ocr.io.loaders import load_images_from_bytes
from caesar_ocr.ocr.tesseract import ocr_tokens_from_image


def _extract_tokens(image: Image.Image, *, lang: str, psm: int) -> Tuple[str, List[Dict[str, object]]]:
    full_text, tokens = ocr_tokens_from_image(image, lang=lang, psm=psm)
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
        source_pdf = rec.get("source_pdf")
        if source_pdf:
            processed.add(source_pdf)
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


def _collect_inputs(input_dir: pathlib.Path, image_exts: List[str], pdf_exts: List[str]) -> List[pathlib.Path]:
    inputs: List[pathlib.Path] = []
    for ext in image_exts:
        inputs.extend(input_dir.glob(f"*.{ext}"))
    for ext in pdf_exts:
        inputs.extend(input_dir.glob(f"*.{ext}"))
    return sorted(set(inputs))


def _filter_inputs(inputs: List[pathlib.Path], patterns: List[str]) -> List[pathlib.Path]:
    if not patterns:
        return inputs
    filtered: List[pathlib.Path] = []
    for path in inputs:
        name = path.name
        if any(fnmatch.fnmatch(name, pat) for pat in patterns):
            filtered.append(path)
    return filtered


def _render_progress(current: int, total: int, *, width: int = 30) -> str:
    if total <= 0:
        return "[no inputs]"
    ratio = min(max(current / total, 0.0), 1.0)
    filled = int(round(ratio * width))
    bar = "#" * filled + "-" * (width - filled)
    return f"[{bar}] {current}/{total} ({ratio * 100:.1f}%)"


def _write_page_image(image: Image.Image, output_dir: pathlib.Path, doc_id: str, page: int) -> pathlib.Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"{doc_id}_page_{page:03d}.png"
    image.save(path)
    return path


def _record_from_tokens(
    *,
    image_path: pathlib.Path,
    doc_id: str,
    page_idx: int,
    full_text: str,
    tokens: List[Dict[str, object]],
    width: int,
    height: int,
    source_pdf: str | None = None,
) -> Tuple[Dict[str, object], Dict[str, object]]:
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
    if source_pdf:
        task["data"]["source_pdf"] = source_pdf

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
    if source_pdf:
        record["source_pdf"] = source_pdf
    return task, record


def main() -> None:
    parser = argparse.ArgumentParser(description="Build LayoutLM JSONL data from images using OCR tokens.")
    parser.add_argument("--input-dir", type=pathlib.Path, required=True, help="Directory with images or PDFs")
    parser.add_argument("--output-dir", type=pathlib.Path, required=True, help="Output directory")
    parser.add_argument("--eval-ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--batch-size", type=int, default=100)
    parser.add_argument("--resume", action="store_true", help="Append to existing outputs and skip processed images")
    parser.add_argument("--lang", default="eng+deu", help="Tesseract language(s), e.g. eng, deu, eng+deu")
    parser.add_argument("--psm", type=int, default=6, help="Tesseract page segmentation mode")
    parser.add_argument("--write-tasks", action="store_true", help="Write Label Studio-style tasks")
    parser.add_argument("--tasks-format", choices=["json", "jsonl"], default="jsonl")
    parser.add_argument("--progress-bar", action="store_true", help="Render a simple progress bar")
    parser.add_argument(
        "--ext",
        action="append",
        default=["png", "jpg", "jpeg", "tif", "tiff"],
        help="Image extension to include (repeatable)",
    )
    parser.add_argument(
        "--pdf-ext",
        action="append",
        default=["pdf"],
        help="PDF extension to include (repeatable)",
    )
    parser.add_argument(
        "--include",
        action="append",
        default=[],
        help="Filename glob to include (repeatable), e.g. '*2025_*.pdf'",
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

    inputs = _collect_inputs(args.input_dir, args.ext, args.pdf_ext)
    inputs = _filter_inputs(inputs, args.include)
    if not inputs:
        raise SystemExit("No images or PDFs found.")

    rng = random.Random(args.seed)
    total = 0
    train_count = 0
    eval_count = 0
    processed = _load_processed_images(all_path) if args.resume else set()
    total_inputs = len(inputs) - len(processed)
    processed_inputs = 0

    tasks: List[Dict[str, object]] = []
    with all_path.open("a", encoding="utf-8") as all_f, \
         train_path.open("a", encoding="utf-8") as train_f, \
         eval_path.open("a", encoding="utf-8") as eval_f:
        for batch in _iter_batches(inputs, args.batch_size):
            for input_path in batch:
                if args.resume and str(input_path) in processed:
                    continue

                if input_path.suffix.lower() == ".pdf":
                    pages = load_images_from_bytes(input_path.read_bytes(), dpi=300)
                    doc_id = input_path.stem
                    for page in pages:
                        page_image_path = _write_page_image(
                            page.image, args.output_dir / "images", doc_id, page.page
                        )
                        full_text, tokens = _extract_tokens(page.image, lang=args.lang, psm=args.psm)
                        task, record = _record_from_tokens(
                            image_path=page_image_path,
                            doc_id=doc_id,
                            page_idx=page.page,
                            full_text=full_text,
                            tokens=tokens,
                            width=page.width,
                            height=page.height,
                            source_pdf=str(input_path),
                        )
                        if args.write_tasks:
                            tasks.append(task)
                        all_f.write(json.dumps(record, ensure_ascii=False) + "\n")
                        if rng.random() < args.eval_ratio:
                            eval_f.write(json.dumps(record, ensure_ascii=False) + "\n")
                            eval_count += 1
                        else:
                            train_f.write(json.dumps(record, ensure_ascii=False) + "\n")
                            train_count += 1
                        total += 1
                    processed_inputs += 1
                else:
                    with Image.open(input_path) as im:
                        image = im.convert("RGB")
                        width, height = image.size
                    full_text, tokens = _extract_tokens(image, lang=args.lang, psm=args.psm)
                    doc_id = input_path.stem
                    page_idx = 1
                    task, record = _record_from_tokens(
                        image_path=input_path,
                        doc_id=doc_id,
                        page_idx=page_idx,
                        full_text=full_text,
                        tokens=tokens,
                        width=width,
                        height=height,
                    )
                    if args.write_tasks:
                        tasks.append(task)
                    all_f.write(json.dumps(record, ensure_ascii=False) + "\n")
                    if rng.random() < args.eval_ratio:
                        eval_f.write(json.dumps(record, ensure_ascii=False) + "\n")
                        eval_count += 1
                    else:
                        train_f.write(json.dumps(record, ensure_ascii=False) + "\n")
                        train_count += 1
                    total += 1
                    processed_inputs += 1

            if args.progress_bar:
                bar = _render_progress(processed_inputs, total_inputs)
                sys.stdout.write(f"\r{bar} records={total}")
                sys.stdout.flush()
            else:
                if total_inputs > 0:
                    pct = (processed_inputs / total_inputs) * 100
                    print(f"Progress: {processed_inputs}/{total_inputs} inputs ({pct:.1f}%), records={total}")
                else:
                    print(f"Processed batch of {len(batch)} inputs (records={total})")

    if args.progress_bar:
        sys.stdout.write("\n")
        sys.stdout.flush()

    if args.write_tasks:
        _write_tasks(tasks_path, tasks, args.tasks_format)
        print(f"Wrote {len(tasks)} tasks to {tasks_path}")
    print(f"Wrote {total} records to {all_path}")
    print(f"Wrote {train_count} train records to {train_path}")
    print(f"Wrote {eval_count} eval records to {eval_path}")


if __name__ == "__main__":
    main()
