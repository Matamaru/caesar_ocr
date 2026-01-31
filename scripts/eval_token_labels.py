"""Evaluate token label predictions against labeled JSONL."""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from caesar_ocr.layoutlm.token_infer import TokenInferer


@dataclass
class LabelStats:
    tp: int = 0
    fp: int = 0
    fn: int = 0

    def precision(self) -> float:
        return self.tp / (self.tp + self.fp) if (self.tp + self.fp) else 0.0

    def recall(self) -> float:
        return self.tp / (self.tp + self.fn) if (self.tp + self.fn) else 0.0

    def f1(self) -> float:
        p = self.precision()
        r = self.recall()
        return (2 * p * r / (p + r)) if (p + r) else 0.0


def _iter_jsonl(path: Path) -> Iterable[Dict]:
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        yield json.loads(line)


def _update_stats(stats: Dict[str, LabelStats], gold: str, pred: str) -> None:
    if gold == pred:
        if gold != "O":
            stats[gold].tp += 1
        return
    if pred != "O":
        stats[pred].fp += 1
    if gold != "O":
        stats[gold].fn += 1


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate LayoutLM token labels against labeled JSONL.")
    parser.add_argument("input", type=Path, help="Labeled JSONL with tokens/bboxes/labels.")
    parser.add_argument("--model-dir", required=True, help="LayoutLM token model directory")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of pages")
    parser.add_argument("--output", type=Path, default=None, help="Write per-page predictions JSONL")
    args = parser.parse_args()

    inferer = TokenInferer.from_model_dir(args.model_dir)
    stats: Dict[str, LabelStats] = defaultdict(LabelStats)
    total_tokens = 0
    per_page_rows = []

    for idx, item in enumerate(_iter_jsonl(args.input)):
        if args.limit is not None and idx >= args.limit:
            break
        image_path = Path(item["image"])
        tokens: List[str] = item.get("tokens", [])
        bboxes: List[List[int]] = item.get("bboxes", [])
        gold_labels: List[str] = item.get("labels", [])
        if not image_path.exists():
            raise FileNotFoundError(f"Missing image: {image_path}")
        if not (len(tokens) == len(bboxes) == len(gold_labels)):
            raise ValueError(f"Token/box/label length mismatch for {image_path}")

        image = Image.open(image_path).convert("RGB")
        pred_labels, _ = inferer.infer(image, tokens, bboxes)
        if len(pred_labels) != len(gold_labels):
            raise ValueError(f"Prediction length mismatch for {image_path}")

        for gold, pred in zip(gold_labels, pred_labels):
            _update_stats(stats, gold, pred)
        total_tokens += len(gold_labels)

        if args.output:
            per_page_rows.append(
                {
                    "doc_id": item.get("doc_id"),
                    "page": item.get("page"),
                    "image": str(image_path),
                    "tokens": tokens,
                    "gold_labels": gold_labels,
                    "pred_labels": pred_labels,
                }
            )

    summary = {}
    for label, stat in sorted(stats.items()):
        summary[label] = {
            "tp": stat.tp,
            "fp": stat.fp,
            "fn": stat.fn,
            "precision": stat.precision(),
            "recall": stat.recall(),
            "f1": stat.f1(),
        }

    print(json.dumps({"total_tokens": total_tokens, "summary": summary}, indent=2))

    if args.output:
        args.output.write_text("\n".join(json.dumps(r) for r in per_page_rows) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
