"""Tests for audio loading and short-chunk segmentation."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from edm_classifier.config import AudioConfig
from edm_classifier.features.audio_io import (
    UnsupportedAudioFormatError,
    is_supported,
    load_audio,
    segment_waveform,
)


def _cfg(**overrides) -> AudioConfig:
    """A deterministic segmentation config, independent of production defaults."""
    base = dict(
        segment_seconds=1.0,
        segment_overlap=0.5,
        trim_intro_seconds=0.0,
        trim_outro_seconds=0.0,
    )
    base.update(overrides)
    return AudioConfig(**base)


def test_is_supported():
    assert is_supported("a.wav")
    assert is_supported("a.MP3")
    assert is_supported("a.aiff")
    assert not is_supported("a.flac")
    assert not is_supported("a.txt")


def test_load_wav(wav_file: Path):
    wav = load_audio(wav_file)
    assert wav.ndim == 1
    assert wav.dtype == np.float32
    assert wav.shape[0] > 0


def test_load_aiff(aiff_file: Path):
    wav = load_audio(aiff_file)
    assert wav.ndim == 1
    assert wav.shape[0] > 0


def test_load_unsupported_extension(tmp_path: Path):
    bad = tmp_path / "track.flac"
    bad.write_bytes(b"not audio")
    with pytest.raises(UnsupportedAudioFormatError):
        load_audio(bad)


def test_load_missing_file(tmp_path: Path):
    with pytest.raises(FileNotFoundError):
        load_audio(tmp_path / "missing.wav")


def test_segment_shapes(waveform: np.ndarray):
    cfg = _cfg()
    segments = segment_waveform(waveform, cfg)  # 3 s / 1 s segments, 50% overlap
    assert segments.ndim == 2
    assert segments.shape[1] == cfg.segment_samples
    assert segments.shape[0] >= 2


def test_segment_short_waveform_is_padded(short_waveform: np.ndarray):
    cfg = _cfg()
    segments = segment_waveform(short_waveform, cfg)  # 0.5 s < 1 s segment
    assert segments.shape == (1, cfg.segment_samples)
    # Tail beyond the original signal must be zero-padded.
    assert np.all(segments[0, short_waveform.shape[0] :] == 0.0)


def test_segment_rejects_multichannel():
    cfg = _cfg()
    stereo = np.zeros((2, cfg.segment_samples), dtype=np.float32)
    with pytest.raises(ValueError):
        segment_waveform(stereo, cfg)


def test_segment_trims_intro_outro(long_waveform: np.ndarray):
    # 40 s waveform: trimming 15 s + 15 s leaves 10 s -> fewer 4 s segments.
    trim_cfg = _cfg(segment_seconds=4.0, trim_intro_seconds=15.0, trim_outro_seconds=15.0)
    no_trim_cfg = _cfg(segment_seconds=4.0)
    trimmed = segment_waveform(long_waveform, trim_cfg)
    full = segment_waveform(long_waveform, no_trim_cfg)
    assert trimmed.shape[0] < full.shape[0]


def test_segment_trim_skipped_when_track_too_short(waveform: np.ndarray):
    # A 3 s track cannot survive a 15 s + 15 s trim, so trimming is skipped.
    cfg = _cfg(segment_seconds=1.0, trim_intro_seconds=15.0, trim_outro_seconds=15.0)
    segments = segment_waveform(waveform, cfg)
    assert segments.shape[0] >= 2
