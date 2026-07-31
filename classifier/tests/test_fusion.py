"""Tests for the v3 late-fusion model."""

from __future__ import annotations

import pytest
import torch

from edm_classifier.config import NUM_CLASSES, settings
from edm_classifier.models.fusion import build_fusion_model

WIN = settings.features.tempogram_win_length
FOURIER_BINS = WIN // 2 + 1
AUTOCORR_BINS = WIN


def test_fusion_forward_shape():
    model = build_fusion_model(n_channels=8, tempo_channels=8).eval()
    mel = torch.randn(2, 1, 128, 173)
    fourier = torch.randn(2, 1, FOURIER_BINS, 173)
    autocorr = torch.randn(2, 1, AUTOCORR_BINS, 173)
    with torch.no_grad():
        logits = model(mel, fourier, autocorr)
    assert logits.shape == (2, NUM_CLASSES)


def test_fusion_handles_variable_frames():
    # Global/adaptive pooling should make the head length-independent.
    model = build_fusion_model(n_channels=8, tempo_channels=8).eval()
    mel = torch.randn(1, 1, 128, 100)
    fourier = torch.randn(1, 1, FOURIER_BINS, 100)
    autocorr = torch.randn(1, 1, AUTOCORR_BINS, 100)
    with torch.no_grad():
        logits = model(mel, fourier, autocorr)
    assert logits.shape == (1, NUM_CLASSES)


def test_fusion_rejects_mismatched_tempogram_bins():
    model = build_fusion_model(n_channels=8, tempo_channels=8).eval()
    mel = torch.randn(1, 1, 128, 173)
    wrong_fourier = torch.randn(1, 1, FOURIER_BINS + 5, 173)  # wrong bin count
    autocorr = torch.randn(1, 1, AUTOCORR_BINS, 173)
    with pytest.raises(RuntimeError):
        model(mel, wrong_fourier, autocorr)
