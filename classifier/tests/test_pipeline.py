"""Tests for the high-level feature pipeline (file -> model input)."""

from __future__ import annotations

from pathlib import Path

from edm_classifier.config import settings
from edm_classifier.features.pipeline import track_to_model_input, track_to_segments

SEG = settings.audio.segment_samples
N_MELS = settings.features.n_mels


def test_track_to_segments(wav_file: Path):
    segments = track_to_segments(wav_file)
    assert segments.ndim == 2
    assert segments.shape[1] == SEG


def test_track_to_model_input_shape(wav_file: Path):
    x = track_to_model_input(wav_file)
    # (n_segments, channels=1, n_mels, n_frames)
    assert x.ndim == 4
    assert x.shape[1] == 1
    assert x.shape[2] == N_MELS
    assert x.shape[0] >= 1
