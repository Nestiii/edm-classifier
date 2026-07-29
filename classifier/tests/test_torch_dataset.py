"""Tests for the PyTorch dataset and segment collation."""

from __future__ import annotations

from pathlib import Path

import torch
from torch.utils.data import DataLoader

from edm_classifier.config import settings
from edm_classifier.data.dataset import index_directory
from edm_classifier.data.torch_dataset import TrackSegmentDataset, collate_segments

N_MELS = settings.features.n_mels


def test_dataset_item_shapes(dataset_dir: Path):
    records = index_directory(dataset_dir)
    ds = TrackSegmentDataset(records)
    assert len(ds) == len(records)
    segments, label = ds[0]
    assert segments.dim() == 4
    assert segments.shape[1] == 1
    assert segments.shape[2] == N_MELS
    assert label.dtype == torch.long


def test_collate_flattens_and_tracks_ids(dataset_dir: Path):
    records = index_directory(dataset_dir)
    ds = TrackSegmentDataset(records)
    loader = DataLoader(ds, batch_size=3, collate_fn=collate_segments)
    segments, labels, track_ids = next(iter(loader))

    # All three tensors share the same leading (segment) dimension.
    assert segments.shape[0] == labels.shape[0] == track_ids.shape[0]
    assert segments.shape[1] == 1 and segments.shape[2] == N_MELS
    # track_ids index into the 3 tracks of this batch.
    assert set(track_ids.tolist()).issubset({0, 1, 2})
