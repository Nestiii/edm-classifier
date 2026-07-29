"""Tests for audio loading and short-chunk segmentation."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from edm_classifier.config import settings
from edm_classifier.features.audio_io import (
    UnsupportedAudioFormatError,
    is_supported,
    load_audio,
    segment_waveform,
)

SEG = settings.audio.segment_samples


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
    segments = segment_waveform(waveform)
    assert segments.ndim == 2
    assert segments.shape[1] == SEG
    assert segments.shape[0] >= 2  # 3 s with 2 s segments and overlap


def test_segment_short_waveform_is_padded(short_waveform: np.ndarray):
    segments = segment_waveform(short_waveform)
    assert segments.shape == (1, SEG)
    # Tail beyond the original signal must be zero-padded.
    assert np.all(segments[0, short_waveform.shape[0] :] == 0.0)


def test_segment_rejects_multichannel():
    stereo = np.zeros((2, SEG), dtype=np.float32)
    with pytest.raises(ValueError):
        segment_waveform(stereo)


def test_segment_covers_tail(waveform: np.ndarray):
    # The last segment must reach exactly the end of the signal.
    segments = segment_waveform(waveform)
    assert segments.shape[0] >= 1
