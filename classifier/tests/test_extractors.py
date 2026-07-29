"""Tests for the Librosa-based feature extractors."""

from __future__ import annotations

import numpy as np

from edm_classifier.config import settings
from edm_classifier.features import extractors as fx

FEAT = settings.features


def test_mel_spectrogram_shape(waveform: np.ndarray):
    mel = fx.mel_spectrogram(waveform)
    assert mel.ndim == 2
    assert mel.shape[0] == FEAT.n_mels
    assert mel.dtype == np.float32


def test_mel_spectrogram_linear_vs_db(waveform: np.ndarray):
    db = fx.mel_spectrogram(waveform, to_db=True)
    linear = fx.mel_spectrogram(waveform, to_db=False)
    # dB output has non-positive values (ref=max); linear power is non-negative.
    assert db.max() <= 0.0 + 1e-4
    assert linear.min() >= 0.0


def test_mfcc_shape(waveform: np.ndarray):
    out = fx.mfcc(waveform)
    assert out.shape[0] == FEAT.n_mfcc


def test_tempogram_shape(waveform: np.ndarray):
    out = fx.tempogram(waveform)
    assert out.ndim == 2
    assert out.shape[1] > 0


def test_spectral_centroid_shape(waveform: np.ndarray):
    out = fx.spectral_centroid(waveform)
    assert out.shape[0] == 1


def test_spectral_rolloff_shape(waveform: np.ndarray):
    out = fx.spectral_rolloff(waveform)
    assert out.shape[0] == 1


def test_zero_crossing_rate_shape(waveform: np.ndarray):
    out = fx.zero_crossing_rate(waveform)
    assert out.shape[0] == 1
    assert np.all(out >= 0.0) and np.all(out <= 1.0)


def test_chroma_shape(waveform: np.ndarray):
    out = fx.chroma(waveform)
    assert out.shape[0] == FEAT.n_chroma


def test_extract_all_returns_every_feature(waveform: np.ndarray):
    fs = fx.extract_all(waveform)
    d = fs.as_dict()
    expected = {
        "mel_spectrogram",
        "mfcc",
        "tempogram",
        "spectral_centroid",
        "spectral_rolloff",
        "zero_crossing_rate",
        "chroma",
    }
    assert set(d) == expected
    assert all(v.ndim == 2 for v in d.values())


def test_summary_vector_is_finite_1d(waveform: np.ndarray):
    fs = fx.extract_all(waveform)
    vec = fx.summary_vector(fs)
    assert vec.ndim == 1
    assert np.all(np.isfinite(vec))
    # mean + std for each of the 7 features.
    expected_len = 2 * sum(v.shape[0] for v in fs.as_dict().values())
    assert vec.shape[0] == expected_len
