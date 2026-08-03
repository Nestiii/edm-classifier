"""End-to-end smoke test for the v3 multi-feature pipeline and fusion training."""

from __future__ import annotations

from pathlib import Path

import torch

from edm_classifier.data.dataset import index_directory
from edm_classifier.data.preprocess_v3 import preprocess_multifeature
from edm_classifier.data.splits import save_split, stratified_split
from edm_classifier.data.torch_dataset import CachedMultiFeatureDataset
from edm_classifier.training.train import TrainConfig
from edm_classifier.training.train_v3 import train_fusion_model


def test_preprocess_multifeature_cache(split_dataset_dir: Path, tmp_path: Path):
    records = index_directory(split_dataset_dir)
    cache = tmp_path / "cache_v3"
    result = preprocess_multifeature(records, cache)

    # One .npy file per track, per feature.
    assert (cache / "mel" / "track_0000.npy").exists()
    assert (cache / "fourier" / "track_0000.npy").exists()
    assert (cache / "autocorr" / "track_0000.npy").exists()
    # A file per track under each feature dir.
    assert len(list((cache / "autocorr").glob("track_*.npy"))) == len(records)
    assert set(result.feature_shapes) == {"mel", "fourier", "autocorr"}
    assert result.n_tracks == len(records)

    ds = CachedMultiFeatureDataset(cache)
    mel, fourier, autocorr, label = ds[0]
    assert mel.shape[0] == 1 and mel.shape[1] == 128
    assert fourier.shape[1] == 193
    assert autocorr.shape[1] == 384
    assert label.dtype == torch.long


def test_train_fusion_end_to_end(split_dataset_dir: Path, tmp_path: Path):
    records = index_directory(split_dataset_dir)
    cache = tmp_path / "cache_v3"
    preprocess_multifeature(records, cache)
    splits = tmp_path / "splits.json"
    save_split(stratified_split(records, seed=1), splits, seed=1)

    out = tmp_path / "run_v3"
    config = TrainConfig(epochs=2, batch_size=4, n_channels=8, num_workers=0, verbose=False)
    report = train_fusion_model(cache, splits, out, config=config, device="cpu")

    assert (out / "model.pt").exists()
    assert (out / "report.json").exists()
    assert 0.0 <= report["test_track"]["accuracy"] <= 1.0
    assert "confusion_matrix" in report["test_track"]
