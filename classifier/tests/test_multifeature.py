"""Tests for aligned multi-feature extraction (mel + two tempograms)."""

from __future__ import annotations

from pathlib import Path

from edm_classifier.config import settings
from edm_classifier.features.multifeature import (
    frames_per_segment,
    track_to_multifeature,
)

AUDIO = settings.audio
FEAT = settings.features


def test_multifeature_keys_and_shapes(wav_file: Path):
    out = track_to_multifeature(wav_file)
    assert set(out) == {"mel", "fourier", "autocorr"}

    seg_frames = frames_per_segment(AUDIO, FEAT)
    n_seg = out["mel"].shape[0]
    # All three features share the segment count, channel dim and frame length.
    for key in ("mel", "fourier", "autocorr"):
        arr = out[key]
        assert arr.shape[0] == n_seg
        assert arr.shape[1] == 1
        assert arr.shape[3] == seg_frames

    # Feature-specific bin counts.
    assert out["mel"].shape[2] == FEAT.n_mels                       # 128
    assert out["fourier"].shape[2] == FEAT.tempogram_win_length // 2 + 1  # 193
    assert out["autocorr"].shape[2] == FEAT.tempogram_win_length    # 384


def test_multifeature_segments_are_aligned(wav_file: Path):
    out = track_to_multifeature(wav_file)
    # Same number of segments across the three features -> they align 1:1.
    assert out["mel"].shape[0] == out["fourier"].shape[0] == out["autocorr"].shape[0]
