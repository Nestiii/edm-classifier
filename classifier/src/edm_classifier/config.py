"""Central configuration for the EDM classifier.

All tunable constants (audio, feature-extraction and dataset parameters) live
here so the rest of the codebase can stay declarative and easy to test.
Values can be overridden via environment variables prefixed with ``EDM_``.
"""

from __future__ import annotations

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

# The eight target subgenres, in a fixed canonical order. The index of a label
# in this tuple is its integer class id used by the model.
SUBGENRES: tuple[str, ...] = (
    "deep house",
    "tech house",
    "melodic techno",
    "progressive",
    "techno peak time",
    "hard techno",
    "minimal/deep tech",
    "trance",
)

NUM_CLASSES: int = len(SUBGENRES)

# File extensions the system is required to support (MP3, AIFF, WAV).
SUPPORTED_EXTENSIONS: tuple[str, ...] = (".mp3", ".aiff", ".aif", ".wav")


class AudioConfig(BaseSettings):
    """Audio loading and short-chunk segmentation parameters."""

    model_config = SettingsConfigDict(env_prefix="EDM_AUDIO_")

    sample_rate: int = 22050
    mono: bool = True
    # Length of each analysed segment, in seconds (Short-chunk CNN uses 2 s).
    segment_seconds: float = 2.0
    # Fraction of overlap between consecutive segments (0.0 = no overlap).
    segment_overlap: float = 0.5

    @property
    def segment_samples(self) -> int:
        """Number of samples in a single segment."""
        return int(round(self.segment_seconds * self.sample_rate))


class FeatureConfig(BaseSettings):
    """Spectral / temporal feature-extraction parameters (Librosa)."""

    model_config = SettingsConfigDict(env_prefix="EDM_FEAT_")

    n_fft: int = 2048
    hop_length: int = 512
    win_length: int | None = None

    # Mel-spectrogram
    n_mels: int = 128
    fmin: float = 20.0
    fmax: float | None = None  # defaults to sample_rate / 2 when None

    # MFCC
    n_mfcc: int = 20

    # Chroma
    n_chroma: int = 12


class DatasetConfig(BaseSettings):
    """Dataset construction and train/validation/test split parameters."""

    model_config = SettingsConfigDict(env_prefix="EDM_DATASET_")

    tracks_per_class: int = 200
    train_ratio: float = 0.70
    val_ratio: float = 0.15
    test_ratio: float = 0.15
    min_bitrate_kbps: int = 128
    seed: int = 42


class Settings(BaseSettings):
    """Top-level settings aggregating every config group."""

    model_config = SettingsConfigDict(env_prefix="EDM_")

    audio: AudioConfig = Field(default_factory=AudioConfig)
    features: FeatureConfig = Field(default_factory=FeatureConfig)
    dataset: DatasetConfig = Field(default_factory=DatasetConfig)


settings = Settings()
