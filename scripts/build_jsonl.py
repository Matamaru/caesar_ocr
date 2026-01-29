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
    """
    Extract OCR tokens from an image using Tesseract.

    :param image: PIL Image
    :param lang: Tesseract language(s)
    :param psm: Tesseract page segmentation mode
    :return: Tuple of full text and list of token dictionaries
    """
    # Use Tesseract OCR to extract tokens and full text
    full_text, tokens = ocr_tokens_from_image(image, lang=lang, psm=psm)
    return full_text, tokens


def _iter_batches(items: List[pathlib.Path], batch_size: int) -> Iterable[List[pathlib.Path]]:
    """
    Yield successive batches of items from the list.

    :param items: List of pathlib.Path items
    :param batch_size: Size of each batch
    :return: Iterable of lists of pathlib.Path items
    """
    for i in range(0, len(items), batch_size):
        yield items[i : i + batch_size]


def _load_processed_images(path: pathlib.Path) -> Set[str]:
    """
    Load a set of processed image or PDF filenames from a JSONL file.

    :param path: Path to the JSONL file
    :return: Set of processed image or PDF filenames
    """
    if not path.exists():
        return set()
    
    # Load processed images or PDFs from existing JSONL
    processed = set()
    # Read each line and extract image and source_pdf fields
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        try:
            # Parse JSON line
            rec = json.loads(line)
        except json.JSONDecodeError:
            continue
        # Extract image and source_pdf fields
        # image = record image path
        # source_pdf = record source PDF path (if any)
        image = rec.get("image")
        if image:
            processed.add(image)
        source_pdf = rec.get("source_pdf")
        if source_pdf:
            processed.add(source_pdf)
    return processed


def _task_key(doc_id: str, page: int, image: str) -> str:
    """
    Generate a unique task key based on document ID, page number, and image path.

    :param doc_id: Document identifier
    :param page: Page number
    :param image: Image path
    :return: Unique task key string
    """
    # format: "{doc_id}::{page}::{image}"
    # use double colons to avoid conflicts with filenames
    return f"{doc_id}::{page}::{image}"


def _write_tasks(path: pathlib.Path, tasks: List[Dict[str, object]], fmt: str) -> None:
    """
    Write tasks to a file in JSON or JSONL format.
    task: List of task dictionaries with "data" and "meta" fields.
    
    :param path: Output file path
    :param tasks: List of task dictionaries
    :param fmt: Output format ("json" or "jsonl")
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    # if format is json, write as a single JSON array
    if fmt == "json":
        path.write_text(json.dumps(tasks, ensure_ascii=False, indent=2))
        return
    with path.open("w", encoding="utf-8") as f:
        for task in tasks:
            f.write(json.dumps(task, ensure_ascii=False) + "\n")


def _collect_inputs(input_dir: pathlib.Path, image_exts: List[str], pdf_exts: List[str]) -> List[pathlib.Path]:
    """
    Collect input files from the input directory based on image and PDF extensions.

    :param input_dir: Directory to search for input files
    :param image_exts: List of image file extensions
    :param pdf_exts: List of PDF file extensions
    :return: Sorted list of unique input file paths
    """
    inputs: List[pathlib.Path] = []
    # Collect image files
    for ext in image_exts:
        inputs.extend(input_dir.glob(f"*.{ext}"))
    # Collect PDF files
    for ext in pdf_exts:
        inputs.extend(input_dir.glob(f"*.{ext}"))
    return sorted(set(inputs))


def _filter_inputs(inputs: List[pathlib.Path], patterns: List[str]) -> List[pathlib.Path]:
    """
    Filter input files based on filename patterns.
    patterns use Unix shell-style wildcards, e.g. '*2025_*.pdf'.
    
    :param inputs: List of input file paths
    :param patterns: List of filename patterns to include
    :return: Filtered list of input file paths
    """
    if not patterns:
        return inputs    
    filtered: List[pathlib.Path] = []
    # Filter inputs based on patterns
    for path in inputs:
        name = path.name
        if any(fnmatch.fnmatch(name, pat) for pat in patterns):
            filtered.append(path)
    return filtered


def _render_progress(current: int, total: int, *, width: int = 30) -> str:
    """
    Render a simple progress bar.

    :param current: Current progress count
    :param total: Total count for completion
    :param width: Width of the progress bar in characters
    :return: Progress bar string
    """
    if total <= 0:
        return "[no inputs]"
    # Calculate progress ratio
    ratio = min(max(current / total, 0.0), 1.0)
    filled = int(round(ratio * width))
    bar = "#" * filled + "-" * (width - filled)
    return f"[{bar}] {current}/{total} ({ratio * 100:.1f}%)"


def _write_page_image(image: Image.Image, output_dir: pathlib.Path, doc_id: str, page: int) -> pathlib.Path:
    """
    Write a page image to the output directory.
    PDFS are split into individual page images.

    :param image: PIL Image object
    :param output_dir: Directory to save the image
    :param doc_id: Document identifier
    :param page: Page number
    :return: Path to the saved image
    """
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
    """
    Create a task and record dictionary from OCR tokens.
    task: for labeling tools
    record: for LayoutLM training
    tokens: list of token dictionaries with "text" and "bbox" fields
    bbox: [x0, y0, x1, y1] coordinates
    labels: labels are used as identifiers for each token, default to "O"
    spans: spans are used for entity annotations, default to empty list
       
    :param image_path: Path to the image file
    :param doc_id: Document identifier
    :param page_idx: Page index
    :param full_text: Full text extracted from the page
    :param tokens: List of token dictionaries
    :param width: Image width
    :param height: Image height
    :param source_pdf: Optional source PDF file path
    :return: Tuple of task and record dictionaries
    """
    # Create task dictionary
    task = {
        "data": {
            "text": full_text,
            "image": str(image_path),
            "doc_id": doc_id,
            "page": page_idx,
            "task_id": _task_key(doc_id, page_idx, str(image_path)), # unique task identifier in format "{doc_id}::{page}::{image}"
        },
        "meta": {
            # Include OCR tokens and image size metadata
            "tokens": tokens,
            "image_size": {"width": width, "height": height},
        },
    }
    # Include source PDF if available
    if source_pdf:
        task["data"]["source_pdf"] = source_pdf

    # Create record dictionary for LayoutLM
    record = {
        "id": None,
        "image": str(image_path),
        "text": full_text,
        "doc_id": doc_id,
        "page": page_idx,
        "tokens": [t["text"] for t in tokens], # token texts, tokens represent OCR tokens with text and bbox
        "bboxes": [t["bbox"] for t in tokens], # bounding boxes for each token
        "labels": ["O"] * len(tokens), # default labels for each token
        "spans": [], # default empty list for entity annotations
    }
    if source_pdf:
        record["source_pdf"] = source_pdf
    return task, record


def main() -> None:
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Build LayoutLM JSONL data from images using OCR tokens.")
    parser.add_argument("--input-dir", type=pathlib.Path, required=True, help="Directory with images or PDFs")
    parser.add_argument("--output-dir", type=pathlib.Path, required=True, help="Output directory")
    # eval-ratio: fraction of data to use for evaluation set
    parser.add_argument("--eval-ratio", type=float, default=0.1)
    # random seed for shuffling/splitting, default 7
    parser.add_argument("--seed", type=int, default=7)
    # batch-size: number of images to process in each batch
    # batches help manage memory usage and provide progress updates
    parser.add_argument("--batch-size", type=int, default=100)
    # resume: if set, append to existing outputs and skip processed images
    parser.add_argument("--resume", action="store_true", help="Append to existing outputs and skip processed images")
    # Tesseract language(s), default "eng+deu"
    parser.add_argument("--lang", default="eng+deu", help="Tesseract language(s), e.g. eng, deu, eng+deu")
    # Tesseract page segmentation mode, default 6 (Assume a single uniform block of text)
    # see https://tesseract-ocr.github.io/tessdoc/ImproveQuality#page-segmentation-method
    parser.add_argument("--psm", type=int, default=6, help="Tesseract page segmentation mode")
    # Write Label Studio-style tasks, if you want to use a labeling tool
    parser.add_argument("--write-tasks", action="store_true", help="Write Label Studio-style tasks")
    # tasks-format: format for writing tasks, either json or jsonl
    parser.add_argument("--tasks-format", choices=["json", "jsonl"], default="jsonl")
    # progress-bar: if set, render a simple progress bar
    parser.add_argument("--progress-bar", action="store_true", help="Render a simple progress bar")
    # ext: image file extensions to include
    # use if you have non-standard image extensions
    parser.add_argument(
        "--ext",
        action="append",
        default=["png", "jpg", "jpeg", "tif", "tiff"],
        help="Image extension to include (repeatable)",
    )
    # pdf-ext: PDF file extensions to include
    parser.add_argument(
        "--pdf-ext",
        action="append",
        default=["pdf"],
        help="PDF extension to include (repeatable)",
    )
    # include: filename glob patterns to include
    parser.add_argument(
        "--include",
        action="append",
        default=[],
        help="Filename glob to include (repeatable), e.g. '*2025_*.pdf'",
    )
    args = parser.parse_args()

    # Prepare output paths
    # tasks.json or tasks.jsonl
    # layoutlm.jsonl
    # layoutlm_train.jsonl
    # layoutlm_eval.jsonl
    args.output_dir.mkdir(parents=True, exist_ok=True)
    tasks_path = args.output_dir / ("tasks.json" if args.tasks_format == "json" else "tasks.jsonl")
    all_path = args.output_dir / "layoutlm.jsonl"
    train_path = args.output_dir / "layoutlm_train.jsonl"
    eval_path = args.output_dir / "layoutlm_eval.jsonl"

    # If not resuming, remove existing output files
    if not args.resume:
        for path in (tasks_path, all_path, train_path, eval_path):
            if path.exists():
                path.unlink()


    # Collect and filter input files
    inputs = _collect_inputs(args.input_dir, args.ext, args.pdf_ext)
    inputs = _filter_inputs(inputs, args.include)
    if not inputs:
        raise SystemExit("No images or PDFs found.")

    # Initialize random number generator for shuffling/splitting 
    # eval-ratio determines fraction of data for eval set
    rng = random.Random(args.seed)
    total = 0
    train_count = 0
    eval_count = 0
    processed = _load_processed_images(all_path) if args.resume else set()
    total_inputs = len(inputs) - len(processed)
    processed_inputs = 0

    # Process inputs in batches
    tasks: List[Dict[str, object]] = []
    with all_path.open("a", encoding="utf-8") as all_f, \
         train_path.open("a", encoding="utf-8") as train_f, \
         eval_path.open("a", encoding="utf-8") as eval_f:
        for batch in _iter_batches(inputs, args.batch_size):
            for input_path in batch:
                if args.resume and str(input_path) in processed:
                    continue

                # Process PDF files
                if input_path.suffix.lower() == ".pdf":
                    pages = load_images_from_bytes(input_path.read_bytes(), dpi=300)
                    doc_id = input_path.stem
                    for page in pages:
                        page_image_path = _write_page_image(
                            page.image, args.output_dir / "images", doc_id, page.page
                        )
                        # Extract tokens and record from the page image
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
                        # all_f is the combined JSONL file
                        if args.write_tasks:
                            tasks.append(task)
                        all_f.write(json.dumps(record, ensure_ascii=False) + "\n")
                        # Split into train/eval based on eval_ratio
                        if rng.random() < args.eval_ratio:
                            eval_f.write(json.dumps(record, ensure_ascii=False) + "\n")
                            eval_count += 1
                        else:
                            train_f.write(json.dumps(record, ensure_ascii=False) + "\n")
                            train_count += 1
                        total += 1
                    processed_inputs += 1

                # Process image files
                else:
                    # preprocess image
                    with Image.open(input_path) as im:
                        image = im.convert("RGB")
                        width, height = image.size
                    # Extract tokens and record from the image
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
                    # all_f is the combined JSONL file
                    # split into train/eval based on eval_ratio
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

            # Update progress
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
        sys.stdout.flush() # flush is needed to ensure the newline is printed after the progress bar

    # Inform user of output files written
    if args.write_tasks:
        _write_tasks(tasks_path, tasks, args.tasks_format)
        print(f"Wrote {len(tasks)} tasks to {tasks_path}")
    print(f"Wrote {total} records to {all_path}")
    print(f"Wrote {train_count} train records to {train_path}")
    print(f"Wrote {eval_count} eval records to {eval_path}")


if __name__ == "__main__":
    main()
