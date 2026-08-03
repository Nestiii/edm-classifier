"""Multi-feature feature-cache preprocessing for the v3 fusion model.

Caches three time-aligned features per segment — mel-spectrogram, Fourier
tempogram, autocorrelation tempogram — as **one small .npy file per track per
feature** (instead of one giant consolidated file). This keeps individual files
small (~10-25 MB) so they upload/download to Google Drive reliably, and each file
stays memory-mappable for fast random access during training.

Cache layout (``cache_dir``):
    mel/track_0000.npy       float16 (n_seg, 1, 128, F)
    fourier/track_0000.npy   float16 (n_seg, 1, 193, F)
    autocorr/track_0000.npy  float16 (n_seg, 1, 384, F)
    ... (one file per track under each feature dir)
    labels.npy      int64 (N,)   per-segment labels, in track order
    track_ids.npy   int64 (N,)   per-segment source track id, in track order
    index.json      per-feature bins + per-track records
"""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from edm_classifier.config import AudioConfig, FeatureConfig, settings
from edm_classifier.data.dataset import TrackRecord
from edm_classifier.data.preprocess import INDEX_FILE, LABELS_FILE, SEGMENT_DTYPE, TRACK_IDS_FILE
from edm_classifier.features.multifeature import track_to_multifeature

FEATURE_NAMES = ("mel", "fourier", "autocorr")


def track_filename(track_id: int) -> str:
    """Per-track cache filename, zero-padded for stable ordering."""
    return f"track_{track_id:04d}.npy"


@dataclass(frozen=True)
class MultiPreprocessResult:
    """Summary of a multi-feature preprocessing run."""

    cache_dir: Path
    n_tracks: int
    n_segments: int
    feature_shapes: dict[str, list[int]]

    def total_gb(self) -> float:
        bytes_ = sum(int(np.prod(s)) for s in self.feature_shapes.values()) * 2
        return bytes_ / 1e9


def preprocess_multifeature(
    records: list[TrackRecord],
    cache_dir: str | Path,
    audio: AudioConfig | None = None,
    feat: FeatureConfig | None = None,
    progress: Callable[[int, int], None] | None = None,
) -> MultiPreprocessResult:
    """Precompute mel + Fourier/autocorr tempograms, one .npy file per track."""
    audio = audio or settings.audio
    feat = feat or settings.features
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    for name in FEATURE_NAMES:
        (cache_dir / name).mkdir(parents=True, exist_ok=True)

    if not records:
        raise ValueError("No records to preprocess.")

    labels: list[int] = []
    track_ids: list[int] = []
    index_tracks: list[dict[str, object]] = []
    total = len(records)
    total_segments = 0
    bins: dict[str, int] = {}
    n_frames: int | None = None

    for track_id, record in enumerate(records):
        feats = track_to_multifeature(record.path, audio, feat)
        n_seg = int(feats["mel"].shape[0])
        for name in FEATURE_NAMES:
            arr = feats[name].astype(SEGMENT_DTYPE)
            if name not in bins:
                bins[name] = int(arr.shape[2])
                n_frames = int(arr.shape[3])
            np.save(cache_dir / name / track_filename(track_id), arr)

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

    feature_shapes = {
        name: [total_segments, 1, bins[name], int(n_frames)] for name in FEATURE_NAMES
    }
    index = {
        "sample_rate": audio.sample_rate,
        "segment_seconds": audio.segment_seconds,
        "n_frames": n_frames,
        "n_tracks": total,
        "n_segments": total_segments,
        "dtype": np.dtype(SEGMENT_DTYPE).name,
        "per_track": True,
        "features": {name: {"dir": name, "bins": bins[name]} for name in FEATURE_NAMES},
        "tracks": index_tracks,
    }
    (cache_dir / INDEX_FILE).write_text(
        json.dumps(index, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    return MultiPreprocessResult(
        cache_dir=cache_dir,
        n_tracks=total,
        n_segments=total_segments,
        feature_shapes=feature_shapes,
    )
