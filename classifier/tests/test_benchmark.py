"""Tests for the per-track processing-time benchmark."""

from __future__ import annotations

from pathlib import Path

import pytest

from edm_classifier.inference.benchmark import BenchmarkResult, benchmark_tracks, find_audio
from edm_classifier.inference.predictor import Predictor
from edm_classifier.models.short_chunk_cnn import build_model


def test_find_audio(dataset_dir: Path):
    # The fixture's tracks live in subfolders; find_audio scans one level.
    genre_dir = next(p for p in dataset_dir.iterdir() if p.is_dir() and any(p.glob("*.wav")))
    files = find_audio(genre_dir)
    assert files
    assert all(f.suffix == ".wav" for f in files)


def test_benchmark_times_each_track(wav_file: Path):
    predictor = Predictor(build_model(n_channels=8), device="cpu")
    result = benchmark_tracks(predictor, [wav_file, wav_file], warmup=1)
    assert isinstance(result, BenchmarkResult)
    assert result.n_tracks == 2
    assert len(result.times) == 2
    assert result.mean > 0
    assert result.maximum >= result.minimum
    # A tiny model on a short clip is well under the 5 s target.
    assert result.meets_target(5.0)
    summary = result.summary()
    assert summary["meets_target"] is True
    assert summary["device"] == "cpu"


def test_benchmark_rejects_empty():
    predictor = Predictor(build_model(n_channels=8), device="cpu")
    with pytest.raises(ValueError):
        benchmark_tracks(predictor, [])
