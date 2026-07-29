"""Tests for track-level inference and prediction aggregation."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from edm_classifier.config import NUM_CLASSES, SUBGENRES
from edm_classifier.inference.predictor import Predictor
from edm_classifier.models.short_chunk_cnn import build_model


def test_prediction_structure_from_probs():
    probs = np.zeros(NUM_CLASSES)
    probs[2] = 0.7
    probs[5] = 0.2
    pred = Predictor._to_prediction(probs)
    assert pred.subgenre == SUBGENRES[2]
    assert pred.confidence == 0.7
    assert pred.label_index == 2
    assert pred.top2[0][0] == SUBGENRES[2]
    assert pred.top2[1][0] == SUBGENRES[5]
    assert len(pred.probabilities) == NUM_CLASSES


def test_predict_track_end_to_end(wav_file: Path):
    predictor = Predictor(build_model(), device="cpu")
    pred = predictor.predict_track(wav_file)
    assert pred.subgenre in SUBGENRES
    assert 0.0 <= pred.confidence <= 1.0
    # Probabilities form a valid distribution.
    total = sum(pred.probabilities.values())
    assert abs(total - 1.0) < 1e-4


def test_checkpoint_roundtrip(tmp_path: Path, wav_file: Path):
    import torch

    model = build_model()
    ckpt = tmp_path / "model.pt"
    torch.save(model.state_dict(), ckpt)

    predictor = Predictor.from_checkpoint(ckpt, device="cpu")
    pred = predictor.predict_track(wav_file)
    assert pred.subgenre in SUBGENRES
