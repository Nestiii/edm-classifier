"""Classical ML baselines over summary audio features.

Provides a reference floor to compare the Short-chunk CNN against — the idea
carried over from the earlier ``ml_models`` prototype, but done honestly: the
scaler is fit on the training split only (no leakage) and splits are stratified.
Each track is reduced to a fixed-length mean/std feature vector.
"""

from __future__ import annotations

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from edm_classifier.config import AudioConfig, FeatureConfig, settings
from edm_classifier.data.dataset import TrackRecord
from edm_classifier.features.audio_io import load_audio, segment_waveform
from edm_classifier.features.extractors import extract_all, summary_vector
from edm_classifier.training.evaluation import EvaluationResult, evaluate


def track_summary_features(
    record: TrackRecord,
    audio: AudioConfig | None = None,
    feat: FeatureConfig | None = None,
) -> np.ndarray:
    """Reduce a track to a single summary feature vector (mean over segments)."""
    audio = audio or settings.audio
    feat = feat or settings.features
    waveform = load_audio(record.path, audio)
    segments = segment_waveform(waveform, audio)
    vectors = [summary_vector(extract_all(seg, audio, feat)) for seg in segments]
    return np.mean(vectors, axis=0).astype(np.float32)


def build_feature_matrix(
    records: list[TrackRecord],
    audio: AudioConfig | None = None,
    feat: FeatureConfig | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Build an (X, y) matrix of summary features for a list of tracks."""
    X = np.stack([track_summary_features(r, audio, feat) for r in records], axis=0)
    y = np.array([r.label for r in records], dtype=int)
    return X, y


def default_baselines() -> dict[str, Pipeline]:
    """Return the set of baseline classifiers, each with a fitted-on-train scaler."""
    return {
        "logistic_regression": Pipeline(
            [("scaler", StandardScaler()), ("clf", LogisticRegression(max_iter=1000))]
        ),
        "svm_rbf": Pipeline(
            [("scaler", StandardScaler()), ("clf", SVC(kernel="rbf", probability=True))]
        ),
        "random_forest": Pipeline(
            [("scaler", StandardScaler()), ("clf", RandomForestClassifier(n_estimators=300))]
        ),
    }


def train_baselines(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_eval: np.ndarray,
    y_eval: np.ndarray,
    models: dict[str, Pipeline] | None = None,
) -> dict[str, EvaluationResult]:
    """Fit each baseline on train and evaluate on the eval split.

    Returns a mapping ``model_name -> EvaluationResult``.
    """
    models = models or default_baselines()
    results: dict[str, EvaluationResult] = {}
    for name, pipeline in models.items():
        pipeline.fit(X_train, y_train)
        probs = pipeline.predict_proba(X_eval)
        results[name] = evaluate(y_eval, probs)
    return results
