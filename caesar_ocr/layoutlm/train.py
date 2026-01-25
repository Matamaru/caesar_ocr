"""LayoutLMv3 training utilities."""

from __future__ import annotations

import json
import pathlib
from dataclasses import dataclass
from typing import Dict, List

from PIL import Image
import torch
from torch.utils.data import Dataset


def read_jsonl(path: pathlib.Path) -> List[Dict[str, object]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def collect_labels(records: List[Dict[str, object]]) -> List[str]:
    labels = set()
    for rec in records:
        for label in rec.get("labels", []):
            labels.add(label)
    labels_list = sorted(labels)
    if "O" not in labels_list:
        labels_list = ["O"] + labels_list
    return labels_list


def normalize_box(box: List[int], width: int, height: int) -> List[int]:
    x0, y0, x1, y1 = box
    return [
        max(0, min(1000, int(1000 * x0 / width))),
        max(0, min(1000, int(1000 * y0 / height))),
        max(0, min(1000, int(1000 * x1 / width))),
        max(0, min(1000, int(1000 * y1 / height))),
    ]


@dataclass
class LayoutLMTokenDataset(Dataset):
    records: List[Dict[str, object]]
    processor: object
    label2id: Dict[str, int]
    max_length: int

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        rec = self.records[idx]
        image_path = pathlib.Path(rec["image"])
        image = Image.open(image_path).convert("RGB")
        width, height = image.size
        tokens = rec["tokens"]
        boxes = rec["bboxes"]
        labels = rec.get("labels", ["O"] * len(tokens))

        norm_boxes = [normalize_box(b, width, height) for b in boxes]
        label_ids = [self.label2id.get(lbl, self.label2id["O"]) for lbl in labels]

        encoding = self.processor(
            images=image,
            text=tokens,
            boxes=norm_boxes,
            word_labels=label_ids,
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        return {k: v.squeeze(0) for k, v in encoding.items()}
