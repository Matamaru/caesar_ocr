"""LayoutLMv3 metrics helpers."""

from __future__ import annotations

from collections import Counter
from typing import Dict, Iterable, List


def precision_recall_f1(y_true: Iterable[str], y_pred: Iterable[str], labels: List[str]) -> Dict[str, Dict[str, float]]:
    metrics: Dict[str, Dict[str, float]] = {}
    true_counts = Counter(y_true)
    pred_counts = Counter(y_pred)

    for label in labels:
        tp = sum(1 for t, p in zip(y_true, y_pred) if t == label and p == label)
        fp = pred_counts[label] - tp
        fn = true_counts[label] - tp
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
        metrics[label] = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "support": float(true_counts[label]),
        }

    return metrics
