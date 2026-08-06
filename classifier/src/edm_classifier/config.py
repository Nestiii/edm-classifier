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

# Filesystem-safe directory name for each subgenre. Datasets are laid out as
# ``<root>/<dirname>/*.wav`` and the desktop app organizes output the same way,
# so slashes/spaces in labels map to a single canonical folder name.
SUBGENRE_DIRNAMES: dict[str, str] = {
    "deep house": "deep_house",
    "tech house": "tech_house",
    "melodic techno": "melodic_techno",
    "progressive": "progressive",
    "techno peak time": "techno_peak_time",
    "hard techno": "hard_techno",
    "minimal/deep tech": "minimal_deep_tech",
    "trance": "trance",
}

# Reverse lookup: directory name -> canonical subgenre label.
DIRNAME_TO_SUBGENRE: dict[str, str] = {v: k for k, v in SUBGENRE_DIRNAMES.items()}

# Alternate on-disk folder names accepted by the indexer, mapped to their
# canonical subgenre. Lets a dataset whose folders differ from the canonical
# dirnames work without renaming (e.g. the legacy "minimal" folder from the
# earlier experiments' dataset stands for "minimal/deep tech").
SUBGENRE_DIRNAME_ALIASES: dict[str, str] = {
    "minimal": "minimal/deep tech",
}


def dirname_subgenre_map() -> dict[str, str]:
    """Directory-name -> subgenre lookup, including known folder-name aliases."""
    return {**DIRNAME_TO_SUBGENRE, **SUBGENRE_DIRNAME_ALIASES}

# Integer class id for each subgenre (index in SUBGENRES).
SUBGENRE_TO_INDEX: dict[str, int] = {name: i for i, name in enumerate(SUBGENRES)}

# File extensions the system is required to support (MP3, AIFF, WAV).
SUPPORTED_EXTENSIONS: tuple[str, ...] = (".mp3", ".aiff", ".aif", ".wav")


def dirname_for(subgenre: str) -> str:
    """Return the filesystem-safe directory name for a subgenre label."""
    return SUBGENRE_DIRNAMES[subgenre]


def subgenre_for_dirname(dirname: str) -> str:
    """Return the canonical subgenre label for a directory name."""
    return DIRNAME_TO_SUBGENRE[dirname]


class AudioConfig(BaseSettings):
    """Audio loading and short-chunk segmentation parameters."""

    model_config = SettingsConfigDict(env_prefix="EDM_AUDIO_")

    sample_rate: int = 22050
    mono: bool = True
    # Length of each analysed segment, in seconds. The plan describes 2 s
    # short-chunks; we use 4 s for more rhythmic context (tunable in WBS 4.5).
    segment_seconds: float = 4.0
    # Fraction of overlap between consecutive segments (0.0 = no overlap).
    segment_overlap: float = 0.5
    # Seconds trimmed from the start/end before segmenting. EDM intros/outros are
    # often unrepresentative (bare kick or silence). Skipped automatically when a
    # track is too short to keep at least one segment after trimming.
    trim_intro_seconds: float = 15.0
    trim_outro_seconds: float = 15.0

    @property
    def segment_samples(self) -> int:
        """Number of samples in a single segment."""
        return int(round(self.segment_seconds * self.sample_rate))

    @property
    def trim_intro_samples(self) -> int:
        return int(round(self.trim_intro_seconds * self.sample_rate))

    @property
    def trim_outro_samples(self) -> int:
        return int(round(self.trim_outro_seconds * self.sample_rate))


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

    # Tempogram (Fourier + autocorrelation) window, in frames. 384 matches the
    # reference paper (Hsu et al. 2021, arXiv:2110.08862) and, at hop 512 /
    # 22.05 kHz, spans ~8.9 s — hence tempograms are computed over the full
    # track and sliced per segment, not per 4 s chunk.
    tempogram_win_length: int = 384


class DatasetConfig(BaseSettings):
    """Dataset construction and train/validation/test split parameters."""

    model_config = SettingsConfigDict(env_prefix="EDM_DATASET_")

    # Actual dataset is 100 tracks/class (the plan's Req 3.1 target was 200).
    tracks_per_class: int = 100
    train_ratio: float = 0.70
    val_ratio: float = 0.15
    test_ratio: float = 0.15
    min_bitrate_kbps: int = 128
    seed: int = 42


class ApiConfig(BaseSettings):
    """Inference API parameters."""

    model_config = SettingsConfigDict(env_prefix="EDM_API_")

    # Path to the trained checkpoint (model.pt). When unset, the API starts but
    # reports the model as not loaded until a path is provided.
    model_path: str | None = None
    device: str = "auto"  # auto | cuda | mps | cpu


class Settings(BaseSettings):
    """Top-level settings aggregating every config group."""

    model_config = SettingsConfigDict(env_prefix="EDM_")

    audio: AudioConfig = Field(default_factory=AudioConfig)
    features: FeatureConfig = Field(default_factory=FeatureConfig)
    dataset: DatasetConfig = Field(default_factory=DatasetConfig)
    api: ApiConfig = Field(default_factory=ApiConfig)


settings = Settings()
