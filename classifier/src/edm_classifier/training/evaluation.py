"""Model evaluation metrics.

Everything here returns plain data (dicts / arrays) rather than printing or
plotting, so results can be logged, asserted in tests and rendered by the UI.
Covers the spec's success metrics: overall accuracy (>80%), top-2 accuracy
(>90%), macro F1 and the confusion matrix.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    top_k_accuracy_score,
)

from edm_classifier.config import NUM_CLASSES, SUBGENRES


def aggregate_segment_probs(
    segment_probs: np.ndarray,
    track_ids: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Average per-segment probabilities into one probability per track.

    This is how a track-level prediction is formed from its 2-second segments.

    Args:
        segment_probs: ``(n_segments, n_class)`` softmax probabilities.
        track_ids: ``(n_segments,)`` id of the source track for each segment.

    Returns:
        (track_probs, unique_track_ids) where ``track_probs`` is
        ``(n_tracks, n_class)`` and rows align with ``unique_track_ids``.
    """
    segment_probs = np.asarray(segment_probs, dtype=np.float64)
    track_ids = np.asarray(track_ids)
    unique_ids = np.unique(track_ids)
    track_probs = np.stack(
        [segment_probs[track_ids == tid].mean(axis=0) for tid in unique_ids], axis=0
    )
    return track_probs.astype(np.float32), unique_ids


@dataclass
class EvaluationResult:
    """Bundle of evaluation metrics for one dataset partition."""

    accuracy: float
    top2_accuracy: float
    macro_f1: float
    per_class_f1: dict[str, float]
    confusion_matrix: np.ndarray = field(repr=False)
    n_samples: int

    def meets_targets(self, accuracy_target: float = 0.80, top2_target: float = 0.90) -> bool:
        """True if accuracy and top-2 accuracy meet the spec thresholds."""
        return self.accuracy >= accuracy_target and self.top2_accuracy >= top2_target

    def summary(self) -> dict[str, float]:
        return {
            "accuracy": round(self.accuracy, 4),
            "top2_accuracy": round(self.top2_accuracy, 4),
            "macro_f1": round(self.macro_f1, 4),
            "n_samples": self.n_samples,
        }


def evaluate(
    y_true: np.ndarray,
    probs: np.ndarray,
    n_class: int = NUM_CLASSES,
) -> EvaluationResult:
    """Compute the full metric bundle from true labels and predicted probs.

    Args:
        y_true: ``(n_samples,)`` integer ground-truth labels.
        probs: ``(n_samples, n_class)`` predicted probabilities.
        n_class: Number of classes.

    Returns:
        An :class:`EvaluationResult`.
    """
    y_true = np.asarray(y_true).astype(int)
    probs = np.asarray(probs, dtype=np.float64)
    if probs.ndim != 2 or probs.shape[1] != n_class:
        raise ValueError(f"Expected probs of shape (n, {n_class}), got {probs.shape}.")
    if y_true.shape[0] != probs.shape[0]:
        raise ValueError("y_true and probs must have the same number of samples.")

    y_pred = probs.argmax(axis=1)
    labels = list(range(n_class))

    acc = float(accuracy_score(y_true, y_pred))
    top2 = float(top_k_accuracy_score(y_true, probs, k=2, labels=labels))
    macro_f1 = float(f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0))
    per_class = f1_score(y_true, y_pred, labels=labels, average=None, zero_division=0)
    per_class_f1 = {SUBGENRES[i]: float(per_class[i]) for i in labels}
    cm = confusion_matrix(y_true, y_pred, labels=labels)

    return EvaluationResult(
        accuracy=acc,
        top2_accuracy=top2,
        macro_f1=macro_f1,
        per_class_f1=per_class_f1,
        confusion_matrix=cm,
        n_samples=int(y_true.shape[0]),
    )
