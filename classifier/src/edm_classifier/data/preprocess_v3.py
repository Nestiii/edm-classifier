"""Multi-feature feature-cache preprocessing for the v3 fusion model.

Like :mod:`edm_classifier.data.preprocess` but caches three time-aligned features
per segment — mel-spectrogram, Fourier tempogram, autocorrelation tempogram — each
streamed to its own raw float16 file (memory-safe). Reuses the same track-level
segmentation as v1/v2 so the persisted split stays comparable.

Cache layout (``cache_dir``):
    mel.f16        float16 (N, 1, 128, F)
    fourier.f16    float16 (N, 1, 193, F)
    autocorr.f16   float16 (N, 1, 384, F)
    labels.npy     int64   (N,)
    track_ids.npy  int64   (N,)
    index.json     per-feature shapes + per-track records
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
FEATURE_FILES = {name: f"{name}.f16" for name in FEATURE_NAMES}


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
    """Precompute mel + Fourier/autocorr tempograms for every track into a cache."""
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
    bins: dict[str, int] = {}
    n_frames: int | None = None

    handles = {name: open(cache_dir / FEATURE_FILES[name], "wb") for name in FEATURE_NAMES}
    try:
        for track_id, record in enumerate(records):
            feats = track_to_multifeature(record.path, audio, feat)
            n_seg = int(feats["mel"].shape[0])
            for name in FEATURE_NAMES:
                arr = feats[name].astype(SEGMENT_DTYPE)
                if name not in bins:
                    bins[name] = int(arr.shape[2])
                    n_frames = int(arr.shape[3])
                arr.tofile(handles[name])

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
    finally:
        for h in handles.values():
            h.close()

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
        "features": {
            name: {"file": FEATURE_FILES[name], "shape": feature_shapes[name]}
            for name in FEATURE_NAMES
        },
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
