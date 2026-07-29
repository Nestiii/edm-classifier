"""Smoke test for the training loop (tiny model, few epochs, CPU)."""

from __future__ import annotations

from pathlib import Path

from edm_classifier.data.dataset import index_directory
from edm_classifier.data.preprocess import preprocess_dataset
from edm_classifier.data.splits import save_split, stratified_split
from edm_classifier.inference.predictor import Predictor
from edm_classifier.training.train import TrainConfig, select_device, train_model


def test_select_device_returns_device():
    dev = select_device("cpu")
    assert dev.type == "cpu"


def test_train_end_to_end(split_dataset_dir: Path, tmp_path: Path):
    records = index_directory(split_dataset_dir)
    cache = tmp_path / "cache"
    preprocess_dataset(records, cache)

    splits = tmp_path / "splits.json"
    save_split(stratified_split(records, seed=1), splits, seed=1)

    out = tmp_path / "run"
    config = TrainConfig(
        epochs=2,
        batch_size=4,
        n_channels=8,  # tiny model for speed
        num_workers=0,
        use_spec_augment=True,
    )
    report = train_model(cache, splits, out, config=config, device="cpu")

    # Checkpoint and report were written.
    assert (out / "model.pt").exists()
    assert (out / "report.json").exists()
    assert report["epochs_trained"] >= 1
    assert 0.0 <= report["test_segment"]["accuracy"] <= 1.0
    assert 0.0 <= report["test_track"]["accuracy"] <= 1.0

    # The saved checkpoint loads back into a working predictor.
    predictor = Predictor.from_checkpoint(out / "model.pt", device="cpu", n_channels=8)
    track = records[0]
    pred = predictor.predict_track(track.path)
    assert 0.0 <= pred.confidence <= 1.0
