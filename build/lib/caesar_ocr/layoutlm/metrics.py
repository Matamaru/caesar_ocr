"""LayoutLMv3 metrics helpers."""

from __future__ import annotations

from collections import Counter
from typing import Dict, Iterable, List


def precision_recall_f1(y_true: Iterable[str], y_pred: Iterable[str], labels: List[str]) -> Dict[str, Dict[str, float]]:
    """
    Compute precision, recall, and F1-score for each label.
    - precision means the ratio of correctly predicted positive observations to the total predicted positive observations.
    - recall means the ratio of correctly predicted positive observations to all observations in actual class.
    - F1 Score is the weighted average of Precision and Recall. Therefore, this score takes both

    :param y_true: True labels.
    :param y_pred: Predicted labels.
    :param labels: List of labels to compute metrics for.
    :return: Dictionary mapping each label to its precision, recall, F1-score, and support.
    """
    # Initialize counters and metrics dictionary
    metrics: Dict[str, Dict[str, float]] = {}
    true_counts = Counter(y_true)
    pred_counts = Counter(y_pred)

    # Calculate precision, recall, and F1-score for each label
    for label in labels:
        # True positives, false positives, and false negatives for the current label
        tp = sum(1 for t, p in zip(y_true, y_pred) if t == label and p == label)
        fp = pred_counts[label] - tp
        fn = true_counts[label] - tp

        # Calculate precision, recall, and F1-score for the current label
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
        
        # Store the metrics for the current label
        metrics[label] = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "support": float(true_counts[label]),   # Number of true instances for this label
        }

    return metrics
