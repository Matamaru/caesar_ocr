"""LayoutLMv3 training utilities."""

from __future__ import annotations

import json
import pathlib
from dataclasses import dataclass
from typing import Dict, List

from PIL import Image
import torch
from torch.utils.data import Dataset

from .utils import normalize_box


def read_jsonl(path: pathlib.Path) -> List[Dict[str, object]]:
    """
    Read a JSON Lines file and return a list of dictionaries.

    :param path: Path to the JSONL file.
    :return: List of dictionaries parsed from the file.
    """
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def collect_labels(records: List[Dict[str, object]]) -> List[str]:
    """
    Collect unique labels from the records.

    :param records: List of records containing labels.
    :return: Sorted list of unique labels with "O" included.
    """
    # Collect unique labels as a set
    labels = set()
    
    # Iterate through records to gather labels
    for rec in records:
        for label in rec.get("labels", []):
            labels.add(label)
    
    # Convert set to sorted list
    labels_list = sorted(labels)

    # Ensure "O" label is included
    if "O" not in labels_list:
        labels_list = ["O"] + labels_list

    return labels_list


@dataclass
class LayoutLMTokenDataset(Dataset):
    """
    Dataset for LayoutLMv3 token classification training.
    
    :param records: List of records containing image paths, tokens, bounding boxes, and labels.
    :param processor: Pretrained LayoutLM processor for tokenization and encoding.
    :param label2id: Dictionary mapping label strings to label IDs.
    :param max_length: Maximum sequence length for the model.
    """

    # Dataset fields
    records: List[Dict[str, object]]
    processor: object
    label2id: Dict[str, int]
    max_length: int

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Get a single dataset item by index.

        :param idx: Index of the item to retrieve.
        :return: Dictionary of tensors for the model input.     
        """

        # Load record and image
        rec = self.records[idx]
        image_path = pathlib.Path(rec["image"])
        image = Image.open(image_path).convert("RGB")
        
        # Normalize bounding boxes and prepare labels
        width, height = image.size
        tokens = rec["tokens"]
        boxes = rec["bboxes"]
        labels = rec.get("labels", ["O"] * len(tokens))

        norm_boxes = [normalize_box(b, width, height) for b in boxes]
        label_ids = [self.label2id.get(lbl, self.label2id["O"]) for lbl in labels]

        # Encode inputs using the processor
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
        # Squeeze batch dimension and return
        return {k: v.squeeze(0) for k, v in encoding.items()}
