"""Tests for dataset indexing and stratified splitting."""

from __future__ import annotations

from pathlib import Path

import pytest

from edm_classifier.data.dataset import (
    assert_min_tracks_per_class,
    class_distribution,
    index_directory,
)
from edm_classifier.data.splits import load_split, save_split, stratified_split


def test_index_directory_finds_labeled_tracks(dataset_dir: Path):
    records = index_directory(dataset_dir)
    assert len(records) == 8  # 2 genres x 4 tracks
    subgenres = {r.subgenre for r in records}
    assert subgenres == {"deep house", "trance"}
    # The unsupported .txt and the unknown folder must be ignored.
    assert all(r.path.suffix == ".wav" for r in records)


def test_index_directory_assigns_correct_labels(dataset_dir: Path):
    records = index_directory(dataset_dir)
    for r in records:
        if r.subgenre == "deep house":
            assert r.label == 0
        elif r.subgenre == "trance":
            assert r.label == 7


def test_index_directory_missing_root(tmp_path: Path):
    with pytest.raises(FileNotFoundError):
        index_directory(tmp_path / "nope")


def test_class_distribution(dataset_dir: Path):
    records = index_directory(dataset_dir)
    dist = class_distribution(records)
    assert dist["deep house"] == 4
    assert dist["trance"] == 4
    assert dist["hard techno"] == 0  # absent genre reported as zero


def test_assert_min_tracks_raises_when_short(dataset_dir: Path):
    # The fixture only has 2 of the 8 subgenres, so any positive minimum fails
    # (the 6 absent subgenres have 0 tracks) — this enforces Req 3.1 fully.
    records = index_directory(dataset_dir)
    with pytest.raises(ValueError):
        assert_min_tracks_per_class(records, minimum=200)
    with pytest.raises(ValueError):
        assert_min_tracks_per_class(records, minimum=1)


def _make_records():
    from edm_classifier.data.dataset import TrackRecord

    records = []
    for label in range(8):
        for i in range(20):
            records.append(
                TrackRecord(path=Path(f"/tmp/g{label}/t{i}.wav"), subgenre=str(label), label=label)
            )
    return records


def test_stratified_split_ratios_and_balance():
    records = _make_records()
    split = stratified_split(records, seed=42)
    sizes = split.sizes()
    assert sizes["train"] + sizes["val"] + sizes["test"] == len(records)
    # 70/15/15 of 160 tracks.
    assert sizes["train"] == pytest.approx(112, abs=2)
    assert sizes["val"] == pytest.approx(24, abs=2)
    assert sizes["test"] == pytest.approx(24, abs=2)

    # No track leaks across partitions.
    paths = lambda recs: {str(r.path) for r in recs}  # noqa: E731
    assert paths(split.train).isdisjoint(paths(split.val))
    assert paths(split.train).isdisjoint(paths(split.test))
    assert paths(split.val).isdisjoint(paths(split.test))


def test_stratified_split_is_reproducible():
    records = _make_records()
    a = stratified_split(records, seed=7)
    b = stratified_split(records, seed=7)
    assert [str(r.path) for r in a.train] == [str(r.path) for r in b.train]


def test_stratified_split_rejects_bad_ratios():
    records = _make_records()
    with pytest.raises(ValueError):
        stratified_split(records, train_ratio=0.5, val_ratio=0.3, test_ratio=0.3)


def test_stratified_split_rejects_empty():
    with pytest.raises(ValueError):
        stratified_split([])


def test_split_persistence_roundtrip(tmp_path: Path):
    records = _make_records()
    split = stratified_split(records, seed=3)
    out = tmp_path / "splits.json"
    save_split(split, out, seed=3)
    assert out.exists()

    loaded = load_split(out)
    assert loaded.sizes() == split.sizes()
    assert [str(r.path) for r in loaded.train] == [str(r.path) for r in split.train]
    assert loaded.train[0].label == split.train[0].label
