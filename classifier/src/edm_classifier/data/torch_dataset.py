"""PyTorch dataset producing 2-second mel-spectrogram segments per track.

Each item is a whole track expanded into its segments, kept together so that
inference can aggregate segment predictions back to a single track-level result.
``collate_segments`` flattens a batch of tracks into a flat batch of segments for
training, while remembering which segments belong to which track.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset, Subset

from edm_classifier.config import AudioConfig, FeatureConfig, settings
from edm_classifier.data.dataset import TrackRecord
from edm_classifier.data.preprocess import (
    LABELS_FILE,
    SEGMENT_DTYPE,
    SEGMENTS_NPY,
    SEGMENTS_RAW,
    TRACK_IDS_FILE,
    load_cache_index,
)
from edm_classifier.data.splits import DataSplit
from edm_classifier.features.pipeline import track_to_model_input


class TrackSegmentDataset(Dataset):
    """Maps a list of tracks to (segments, label) tensors.

    Item ``i`` returns:
        segments: float tensor ``(n_segments, 1, n_mels, n_frames)``
        label: long scalar tensor
    """

    def __init__(
        self,
        records: list[TrackRecord],
        audio: AudioConfig | None = None,
        feat: FeatureConfig | None = None,
    ) -> None:
        self.records = records
        self.audio = audio or settings.audio
        self.feat = feat or settings.features

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        record = self.records[idx]
        segments = track_to_model_input(record.path, self.audio, self.feat)
        x = torch.from_numpy(np.ascontiguousarray(segments)).float()
        y = torch.tensor(record.label, dtype=torch.long)
        return x, y


class CachedSegmentDataset(Dataset):
    """Segment-level dataset backed by a precomputed feature cache.

    Item ``i`` returns:
        segment: float tensor ``(1, n_mels, n_frames)``
        label: long scalar tensor

    Segments are memory-mapped, so random access is cheap and the whole array is
    never forced into RAM at once. ``track_ids`` lets a split select segments by
    source track and lets inference aggregate predictions per track.
    """

    def __init__(self, cache_dir: str | Path) -> None:
        self.cache_dir = Path(cache_dir)
        self.index = load_cache_index(self.cache_dir)
        self.segments = self._open_segments()
        self.labels = np.load(self.cache_dir / LABELS_FILE)
        self.track_ids = np.load(self.cache_dir / TRACK_IDS_FILE)
        # Map each track's absolute path to its stable track id.
        self._path_to_track_id = {
            str(t["path"]): int(t["track_id"]) for t in self.index["tracks"]
        }

    def _open_segments(self) -> np.ndarray:
        """Memory-map the segment array (raw float16 preferred, legacy .npy)."""
        raw = self.cache_dir / SEGMENTS_RAW
        if raw.exists():
            shape = tuple(self.index["segment_shape"])
            return np.memmap(raw, dtype=SEGMENT_DTYPE, mode="r", shape=shape)
        legacy = self.cache_dir / SEGMENTS_NPY
        if legacy.exists():
            return np.load(legacy, mmap_mode="r")
        raise FileNotFoundError(
            f"No segment cache ({SEGMENTS_RAW} or {SEGMENTS_NPY}) in {self.cache_dir}."
        )

    def __len__(self) -> int:
        return int(self.segments.shape[0])

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        segment = np.asarray(self.segments[idx], dtype=np.float32)
        x = torch.from_numpy(segment)
        y = torch.tensor(int(self.labels[idx]), dtype=torch.long)
        return x, y

    def segment_indices_for_paths(self, paths: list[str | Path]) -> list[int]:
        """Return segment indices whose source track is in ``paths``."""
        wanted = {self._path_to_track_id[str(p)] for p in paths if str(p) in self._path_to_track_id}
        return [i for i, tid in enumerate(self.track_ids) if int(tid) in wanted]


def build_split_subsets(
    cache_dir: str | Path, split: DataSplit
) -> tuple[Subset, Subset, Subset]:
    """Build train/val/test segment-level subsets from a track-level split.

    Because the split is by track, no segment of the same track appears in more
    than one partition.
    """
    dataset = CachedSegmentDataset(cache_dir)
    train = Subset(dataset, dataset.segment_indices_for_paths([r.path for r in split.train]))
    val = Subset(dataset, dataset.segment_indices_for_paths([r.path for r in split.val]))
    test = Subset(dataset, dataset.segment_indices_for_paths([r.path for r in split.test]))
    return train, val, test


def collate_segments(
    batch: list[tuple[torch.Tensor, torch.Tensor]],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Flatten a batch of per-track segments into a flat segment batch.

    Returns:
        segments: ``(total_segments, 1, n_mels, n_frames)``
        labels: ``(total_segments,)`` — each segment carries its track's label
        track_ids: ``(total_segments,)`` — index of the source track within the
            batch, so predictions can be aggregated per track.
    """
    seg_tensors: list[torch.Tensor] = []
    labels: list[torch.Tensor] = []
    track_ids: list[torch.Tensor] = []
    for track_idx, (segments, label) in enumerate(batch):
        n = segments.shape[0]
        seg_tensors.append(segments)
        labels.append(label.repeat(n))
        track_ids.append(torch.full((n,), track_idx, dtype=torch.long))
    return (
        torch.cat(seg_tensors, dim=0),
        torch.cat(labels, dim=0),
        torch.cat(track_ids, dim=0),
    )
