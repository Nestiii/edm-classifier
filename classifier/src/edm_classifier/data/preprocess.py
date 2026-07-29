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

SEGMENTS_FILE = "segments.npy"
LABELS_FILE = "labels.npy"
TRACK_IDS_FILE = "track_ids.npy"
INDEX_FILE = "index.json"


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

    seg_arrays: list[np.ndarray] = []
    labels: list[int] = []
    track_ids: list[int] = []
    index_tracks: list[dict[str, object]] = []

    total = len(records)
    for track_id, record in enumerate(records):
        segments = track_to_model_input(record.path, audio, feat).astype(np.float16)
        n_seg = segments.shape[0]
        seg_arrays.append(segments)
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

    if not seg_arrays:
        raise ValueError("No records to preprocess.")

    segments = np.concatenate(seg_arrays, axis=0)
    labels_arr = np.asarray(labels, dtype=np.int64)
    track_ids_arr = np.asarray(track_ids, dtype=np.int64)

    np.save(cache_dir / SEGMENTS_FILE, segments)
    np.save(cache_dir / LABELS_FILE, labels_arr)
    np.save(cache_dir / TRACK_IDS_FILE, track_ids_arr)

    n_mels, n_frames = segments.shape[2], segments.shape[3]
    index = {
        "sample_rate": audio.sample_rate,
        "segment_seconds": audio.segment_seconds,
        "n_mels": n_mels,
        "n_frames": n_frames,
        "n_tracks": total,
        "n_segments": int(segments.shape[0]),
        "tracks": index_tracks,
    }
    (cache_dir / INDEX_FILE).write_text(
        json.dumps(index, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    return PreprocessResult(
        cache_dir=cache_dir,
        n_tracks=total,
        n_segments=int(segments.shape[0]),
        n_mels=n_mels,
        n_frames=n_frames,
    )


def load_cache_index(cache_dir: str | Path) -> dict[str, object]:
    """Read the cache ``index.json`` metadata."""
    cache_dir = Path(cache_dir)
    index_path = cache_dir / INDEX_FILE
    if not index_path.exists():
        raise FileNotFoundError(f"No feature cache index at {index_path}.")
    return json.loads(index_path.read_text(encoding="utf-8"))
