"""Dataset manifest: label every track and record its audio metadata.

The manifest is the single source of truth for what the dataset contains. It is
built by scanning the on-disk layout (``<root>/<subgenre>/*.{mp3,aiff,wav}``),
probing each file's audio properties, and is persisted as a CSV so validation,
preprocessing and splitting all read the same table.

Two optional columns, ``source_1``/``source_2``, hold the professional sources
that validated each track's subgenre (Req 3.2); they are left blank when built
from files and can be filled in by hand.
"""

from __future__ import annotations

import csv
from dataclasses import asdict, dataclass
from pathlib import Path

import audioread
import soundfile as sf

from edm_classifier.data.dataset import TrackRecord, index_directory


@dataclass(frozen=True)
class AudioMetadata:
    """Probed audio properties for a single file."""

    sample_rate: int
    channels: int
    duration_seconds: float
    format: str
    bitrate_kbps: float
    size_bytes: int


@dataclass(frozen=True)
class ManifestEntry:
    """A labeled track plus its audio metadata and validation sources."""

    path: str
    subgenre: str
    label: int
    format: str
    sample_rate: int
    channels: int
    duration_seconds: float
    bitrate_kbps: float
    size_bytes: int
    source_1: str = ""
    source_2: str = ""


CSV_FIELDS: list[str] = list(ManifestEntry.__dataclass_fields__.keys())


def probe_audio(path: str | Path) -> AudioMetadata:
    """Read audio metadata without decoding the whole file when possible.

    Falls back to :mod:`audioread` (ffmpeg/etc.) when libsndfile cannot open the
    container (e.g. some MP3s). Bitrate is estimated from file size and duration,
    which is exact for constant-bitrate lossy files and comfortably above the
    128 kbps floor for lossless PCM.
    """
    path = Path(path)
    size_bytes = path.stat().st_size
    try:
        info = sf.info(path)
        sample_rate, duration, channels, fmt = (
            info.samplerate,
            info.duration,
            info.channels,
            info.format,
        )
    except Exception:
        with audioread.audio_open(str(path)) as f:
            sample_rate, duration, channels = f.samplerate, f.duration, f.channels
        fmt = path.suffix.lstrip(".").upper()

    bitrate_kbps = (size_bytes * 8) / duration / 1000 if duration > 0 else 0.0
    return AudioMetadata(
        sample_rate=int(sample_rate),
        channels=int(channels),
        duration_seconds=float(duration),
        format=str(fmt),
        bitrate_kbps=float(bitrate_kbps),
        size_bytes=int(size_bytes),
    )


def _entry_from_record(record: TrackRecord) -> ManifestEntry:
    meta = probe_audio(record.path)
    return ManifestEntry(
        path=str(record.path),
        subgenre=record.subgenre,
        label=record.label,
        format=meta.format,
        sample_rate=meta.sample_rate,
        channels=meta.channels,
        duration_seconds=round(meta.duration_seconds, 3),
        bitrate_kbps=round(meta.bitrate_kbps, 1),
        size_bytes=meta.size_bytes,
    )


def build_manifest(root: str | Path) -> list[ManifestEntry]:
    """Scan a dataset root and build a manifest entry for every track."""
    records = index_directory(root)
    return [_entry_from_record(r) for r in records]


def save_manifest(entries: list[ManifestEntry], path: str | Path) -> None:
    """Write a manifest to a CSV file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        writer.writeheader()
        for entry in entries:
            writer.writerow(asdict(entry))


def _coerce(row: dict[str, str]) -> ManifestEntry:
    return ManifestEntry(
        path=row["path"],
        subgenre=row["subgenre"],
        label=int(row["label"]),
        format=row["format"],
        sample_rate=int(row["sample_rate"]),
        channels=int(row["channels"]),
        duration_seconds=float(row["duration_seconds"]),
        bitrate_kbps=float(row["bitrate_kbps"]),
        size_bytes=int(row["size_bytes"]),
        source_1=row.get("source_1", ""),
        source_2=row.get("source_2", ""),
    )


def load_manifest(path: str | Path) -> list[ManifestEntry]:
    """Read a manifest back from a CSV file."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Manifest not found: {path}")
    with path.open(newline="", encoding="utf-8") as f:
        return [_coerce(row) for row in csv.DictReader(f)]


def to_track_records(entries: list[ManifestEntry]) -> list[TrackRecord]:
    """Convert manifest entries back into lightweight track records."""
    return [
        TrackRecord(path=Path(e.path), subgenre=e.subgenre, label=e.label) for e in entries
    ]
