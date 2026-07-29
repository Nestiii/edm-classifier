"""Tests for feature-cache preprocessing and the cached dataset."""

from __future__ import annotations

from pathlib import Path

import torch

from edm_classifier.config import settings
from edm_classifier.data.dataset import index_directory
from edm_classifier.data.preprocess import (
    INDEX_FILE,
    SEGMENTS_FILE,
    load_cache_index,
    preprocess_dataset,
)
from edm_classifier.data.splits import stratified_split
from edm_classifier.data.torch_dataset import (
    CachedSegmentDataset,
    build_split_subsets,
)

N_MELS = settings.features.n_mels


def test_preprocess_writes_cache(dataset_dir: Path, tmp_path: Path):
    records = index_directory(dataset_dir)
    cache = tmp_path / "cache"
    result = preprocess_dataset(records, cache)

    assert (cache / SEGMENTS_FILE).exists()
    assert (cache / INDEX_FILE).exists()
    assert result.n_tracks == len(records)
    assert result.n_segments >= len(records)  # >= 1 segment per track
    assert result.n_mels == N_MELS

    index = load_cache_index(cache)
    assert index["n_tracks"] == len(records)
    assert len(index["tracks"]) == len(records)


def test_cached_dataset_items(dataset_dir: Path, tmp_path: Path):
    records = index_directory(dataset_dir)
    cache = tmp_path / "cache"
    preprocess_dataset(records, cache)

    ds = CachedSegmentDataset(cache)
    assert len(ds) >= len(records)
    x, y = ds[0]
    assert x.shape[0] == 1 and x.shape[1] == N_MELS
    assert y.dtype == torch.long


def test_build_split_subsets_are_disjoint(split_dataset_dir: Path, tmp_path: Path):
    records = index_directory(split_dataset_dir)
    cache = tmp_path / "cache"
    preprocess_dataset(records, cache)
    split = stratified_split(records, seed=1)

    train, val, test = build_split_subsets(cache, split)
    total = len(train) + len(val) + len(test)
    assert total == len(CachedSegmentDataset(cache))

    # Underlying segment indices must not overlap across partitions.
    idx_train, idx_val, idx_test = set(train.indices), set(val.indices), set(test.indices)
    assert idx_train.isdisjoint(idx_val)
    assert idx_train.isdisjoint(idx_test)
    assert idx_val.isdisjoint(idx_test)


def test_segment_indices_for_paths(dataset_dir: Path, tmp_path: Path):
    records = index_directory(dataset_dir)
    cache = tmp_path / "cache"
    preprocess_dataset(records, cache)
    ds = CachedSegmentDataset(cache)

    some_paths = [records[0].path, records[1].path]
    indices = ds.segment_indices_for_paths(some_paths)
    assert len(indices) >= 2
    # Every returned index really belongs to one of the requested tracks.
    wanted_ids = {ds._path_to_track_id[str(p)] for p in some_paths}
    assert all(int(ds.track_ids[i]) in wanted_ids for i in indices)
