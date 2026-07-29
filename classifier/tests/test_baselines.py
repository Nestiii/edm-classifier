"""Smoke tests for the classical ML baselines."""

from __future__ import annotations

import numpy as np

from edm_classifier.config import NUM_CLASSES
from edm_classifier.training.baselines import default_baselines, train_baselines


def test_train_baselines_returns_metrics_per_model():
    rng = np.random.default_rng(0)
    n_features = 40
    # Separable-ish synthetic data: class-dependent mean shift.
    def make(n_per_class: int):
        X, y = [], []
        for c in range(NUM_CLASSES):
            X.append(rng.normal(loc=c, scale=0.5, size=(n_per_class, n_features)))
            y.extend([c] * n_per_class)
        return np.vstack(X), np.array(y)

    X_train, y_train = make(20)
    X_eval, y_eval = make(6)

    results = train_baselines(X_train, y_train, X_eval, y_eval)
    assert set(results) == set(default_baselines())
    for _name, res in results.items():
        assert 0.0 <= res.accuracy <= 1.0
        assert res.confusion_matrix.shape == (NUM_CLASSES, NUM_CLASSES)
        # With a clear mean shift, models should beat random chance (1/8).
        assert res.accuracy > 1.0 / NUM_CLASSES
