"""Shared pytest fixtures: synthetic audio files and waveforms."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from edm_classifier.config import settings

SR = settings.audio.sample_rate


def _sine(duration_s: float, freq: float = 220.0, sr: int = SR) -> np.ndarray:
    t = np.linspace(0.0, duration_s, int(sr * duration_s), endpoint=False)
    return (0.5 * np.sin(2 * np.pi * freq * t)).astype(np.float32)


@pytest.fixture
def waveform() -> np.ndarray:
    """A 3-second mono sine waveform at the configured sample rate."""
    return _sine(3.0)


@pytest.fixture
def short_waveform() -> np.ndarray:
    """A waveform shorter than a single 2-second segment."""
    return _sine(0.5)


@pytest.fixture
def wav_file(tmp_path: Path) -> Path:
    """A 3-second WAV file on disk."""
    path = tmp_path / "track.wav"
    sf.write(path, _sine(3.0), SR, subtype="PCM_16")
    return path


@pytest.fixture
def aiff_file(tmp_path: Path) -> Path:
    """A 3-second AIFF file on disk."""
    path = tmp_path / "track.aiff"
    sf.write(path, _sine(3.0), SR, format="AIFF", subtype="PCM_16")
    return path


@pytest.fixture
def dataset_dir(tmp_path: Path) -> Path:
    """A small on-disk dataset: 2 subgenres, 4 tracks each, plus noise dirs."""
    from edm_classifier.config import SUBGENRE_DIRNAMES

    root = tmp_path / "dataset"
    dirnames = [SUBGENRE_DIRNAMES["deep house"], SUBGENRE_DIRNAMES["trance"]]
    for i, dirname in enumerate(dirnames):
        genre_dir = root / dirname
        genre_dir.mkdir(parents=True)
        for j in range(4):
            freq = 110.0 * (i + 1) + j
            sf.write(genre_dir / f"track_{j}.wav", _sine(3.0, freq), SR, subtype="PCM_16")

    # Directories/files that must be ignored by the indexer.
    (root / "not_a_genre").mkdir()
    (root / SUBGENRE_DIRNAMES["deep house"] / "cover.txt").write_text("art")  # unsupported
    return root
