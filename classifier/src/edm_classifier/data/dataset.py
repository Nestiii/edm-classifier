"""Dataset indexing: discover labeled audio tracks on disk.

Expected on-disk layout (one subfolder per subgenre, filesystem-safe names):

    <root>/
        deep_house/*.wav
        tech_house/*.mp3
        ...
        minimal_deep_tech/*.aiff
        trance/*.wav
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from pathlib import Path

from edm_classifier.config import (
    DIRNAME_TO_SUBGENRE,
    SUBGENRE_TO_INDEX,
    settings,
)
from edm_classifier.features.audio_io import is_supported


@dataclass(frozen=True)
class TrackRecord:
    """A single labeled audio file in the dataset."""

    path: Path
    subgenre: str
    label: int  # integer class id (index in SUBGENRES)


def index_directory(root: str | Path) -> list[TrackRecord]:
    """Scan a dataset root and return one :class:`TrackRecord` per audio file.

    Subfolders whose name is not a known subgenre directory are ignored, as are
    files with unsupported extensions.

    Args:
        root: Dataset root containing one subfolder per subgenre.

    Returns:
        Records sorted by (subgenre, path) for deterministic ordering.

    Raises:
        FileNotFoundError: If ``root`` does not exist or is not a directory.
    """
    root = Path(root)
    if not root.is_dir():
        raise FileNotFoundError(f"Dataset root not found or not a directory: {root}")

    records: list[TrackRecord] = []
    for dirname, subgenre in DIRNAME_TO_SUBGENRE.items():
        subdir = root / dirname
        if not subdir.is_dir():
            continue
        for path in sorted(subdir.iterdir()):
            if path.is_file() and is_supported(path):
                records.append(
                    TrackRecord(
                        path=path,
                        subgenre=subgenre,
                        label=SUBGENRE_TO_INDEX[subgenre],
                    )
                )
    records.sort(key=lambda r: (r.subgenre, str(r.path)))
    return records


def class_distribution(records: list[TrackRecord]) -> dict[str, int]:
    """Count how many tracks belong to each subgenre."""
    counts = Counter(r.subgenre for r in records)
    return {name: counts.get(name, 0) for name in SUBGENRE_TO_INDEX}


def assert_min_tracks_per_class(
    records: list[TrackRecord],
    minimum: int | None = None,
) -> None:
    """Raise if any subgenre has fewer than ``minimum`` tracks (Req 3.1: 200)."""
    minimum = minimum if minimum is not None else settings.dataset.tracks_per_class
    distribution = class_distribution(records)
    short = {name: n for name, n in distribution.items() if n < minimum}
    if short:
        raise ValueError(
            f"Subgenres below the required {minimum} tracks/class: {short}"
        )
