"""Tests for evaluation metrics and per-track aggregation."""

from __future__ import annotations

import numpy as np

from edm_classifier.config import NUM_CLASSES
from edm_classifier.training.evaluation import aggregate_segment_probs, evaluate


def _one_hot(labels: np.ndarray, n: int = NUM_CLASSES) -> np.ndarray:
    probs = np.full((labels.shape[0], n), 0.01, dtype=np.float64)
    probs[np.arange(labels.shape[0]), labels] = 0.93
    return probs


def test_evaluate_perfect_predictions():
    y = np.array([0, 1, 2, 3, 4, 5, 6, 7])
    probs = _one_hot(y)
    result = evaluate(y, probs)
    assert result.accuracy == 1.0
    assert result.top2_accuracy == 1.0
    assert result.macro_f1 == 1.0
    assert result.meets_targets()
    assert result.confusion_matrix.shape == (NUM_CLASSES, NUM_CLASSES)
    assert result.n_samples == 8


def test_evaluate_top2_counts_second_choice():
    # True class is always the 2nd most probable -> top1 fails, top2 succeeds.
    y = np.array([0, 1, 2, 3])
    probs = np.full((4, NUM_CLASSES), 0.01)
    for i, label in enumerate(y):
        other = (label + 1) % NUM_CLASSES
        probs[i, other] = 0.6   # highest
        probs[i, label] = 0.3   # second highest
    result = evaluate(y, probs)
    assert result.accuracy == 0.0
    assert result.top2_accuracy == 1.0


def test_evaluate_rejects_bad_shape():
    y = np.array([0, 1])
    bad = np.zeros((2, NUM_CLASSES + 1))
    import pytest

    with pytest.raises(ValueError):
        evaluate(y, bad)


def test_aggregate_segment_probs_means_per_track():
    # 3 segments for track 0, 2 for track 1.
    seg_probs = np.array(
        [
            [0.8, 0.2],
            [0.6, 0.4],
            [0.7, 0.3],
            [0.1, 0.9],
            [0.3, 0.7],
        ]
    )
    track_ids = np.array([0, 0, 0, 1, 1])
    track_probs, ids = aggregate_segment_probs(seg_probs, track_ids)
    assert track_probs.shape == (2, 2)
    np.testing.assert_allclose(track_probs[0], [0.7, 0.3], atol=1e-6)
    np.testing.assert_allclose(track_probs[1], [0.2, 0.8], atol=1e-6)
    assert list(ids) == [0, 1]
