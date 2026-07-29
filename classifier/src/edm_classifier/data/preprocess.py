"""Feature precomputation: turn raw audio into a compact on-disk cache.

Extracting mel-spectrograms on every training epoch is wasteful. Instead we run
the feature pipeline once and store all segments as a single consolidated array
(float16 -> roughly 1 GB for the full dataset, small enough to fit in RAM). The
cache is the input to training; raw audio is never read again.

Cache layout (``cache_dir``):
    segments.npy   float16 (total_segments, 1, n_mels, n_frames)
    labels.npy     int64   (total_segments,)
    track_ids.npy  int64   (total_segments,)   -> groups segments by source track
    index.json     metadata + per-track records
"""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from edm_classifier.config import AudioConfig, FeatureConfig, settings
from edm_classifier.data.dataset import TrackRecord
from edm_classifier.features.pipeline import track_to_model_input

# Segments are streamed to a raw float16 file (memory-safe, see below).
# ``SEGMENTS_NPY`` is the legacy consolidated array kept only for backward-compat
# reading of caches produced by older versions.
SEGMENTS_RAW = "segments.f16"
SEGMENTS_NPY = "segments.npy"
LABELS_FILE = "labels.npy"
TRACK_IDS_FILE = "track_ids.npy"
INDEX_FILE = "index.json"
SEGMENT_DTYPE = np.float16


@dataclass(frozen=True)
class PreprocessResult:
    """Summary of a preprocessing run."""

    cache_dir: Path
    n_tracks: int
    n_segments: int
    n_mels: int
    n_frames: int


def preprocess_dataset(
    records: list[TrackRecord],
    cache_dir: str | Path,
    audio: AudioConfig | None = None,
    feat: FeatureConfig | None = None,
    progress: Callable[[int, int], None] | None = None,
) -> PreprocessResult:
    """Precompute mel-spectrogram segments for every track into a cache.

    Args:
        records: Tracks to preprocess (order defines the stable track ids).
        cache_dir: Output directory for the cache files.
        audio, feat: Configs; default to the global settings.
        progress: Optional callback ``(done, total)`` for UI/CLI progress.

    Returns:
        A :class:`PreprocessResult` describing the cache.
    """
    audio = audio or settings.audio
    feat = feat or settings.features
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    if not records:
        raise ValueError("No records to preprocess.")

    labels: list[int] = []
    track_ids: list[int] = []
    index_tracks: list[dict[str, object]] = []
    total = len(records)
    total_segments = 0
    n_mels: int | None = None
    n_frames: int | None = None

    # Stream each track's segments straight to disk (C-order float16). Peak
    # memory is one track's segments, not the whole dataset — avoids the OOM
    # that a concatenate-everything-then-save approach hits on large datasets.
    raw_path = cache_dir / SEGMENTS_RAW
    with open(raw_path, "wb") as fh:
        for track_id, record in enumerate(records):
            segments = track_to_model_input(record.path, audio, feat).astype(SEGMENT_DTYPE)
            if n_mels is None:
                n_mels, n_frames = int(segments.shape[2]), int(segments.shape[3])
            elif (segments.shape[2], segments.shape[3]) != (n_mels, n_frames):
                raise ValueError(
                    f"Inconsistent segment shape for {record.path}: "
                    f"got {segments.shape[2:]}, expected {(n_mels, n_frames)}."
                )
            segments.tofile(fh)  # appends raw C-order bytes

            n_seg = int(segments.shape[0])
            total_segments += n_seg
            labels.extend([record.label] * n_seg)
            track_ids.extend([track_id] * n_seg)
            index_tracks.append(
                {
                    "track_id": track_id,
                    "path": str(record.path),
                    "subgenre": record.subgenre,
                    "label": record.label,
                    "n_segments": n_seg,
                }
            )
            if progress is not None:
                progress(track_id + 1, total)

    np.save(cache_dir / LABELS_FILE, np.asarray(labels, dtype=np.int64))
    np.save(cache_dir / TRACK_IDS_FILE, np.asarray(track_ids, dtype=np.int64))

    segment_shape = [total_segments, 1, n_mels, n_frames]
    index = {
        "sample_rate": audio.sample_rate,
        "segment_seconds": audio.segment_seconds,
        "n_mels": n_mels,
        "n_frames": n_frames,
        "n_tracks": total,
        "n_segments": total_segments,
        "segment_shape": segment_shape,
        "dtype": np.dtype(SEGMENT_DTYPE).name,
        "tracks": index_tracks,
    }
    (cache_dir / INDEX_FILE).write_text(
        json.dumps(index, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    return PreprocessResult(
        cache_dir=cache_dir,
        n_tracks=total,
        n_segments=total_segments,
        n_mels=int(n_mels),
        n_frames=int(n_frames),
    )


def load_cache_index(cache_dir: str | Path) -> dict[str, object]:
    """Read the cache ``index.json`` metadata."""
    cache_dir = Path(cache_dir)
    index_path = cache_dir / INDEX_FILE
    if not index_path.exists():
        raise FileNotFoundError(f"No feature cache index at {index_path}.")
    return json.loads(index_path.read_text(encoding="utf-8"))
