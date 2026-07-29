"""Tests for the command-line interface and the data pipeline it drives."""

from __future__ import annotations

from pathlib import Path

from edm_classifier.cli import main


def test_list_subgenres(capsys):
    assert main(["--list-subgenres"]) == 0
    out = capsys.readouterr().out
    assert "deep house" in out
    assert "trance" in out


def test_no_command_prints_help(capsys):
    assert main([]) == 0
    assert "usage" in capsys.readouterr().out.lower()


def test_data_pipeline_end_to_end(split_dataset_dir: Path, tmp_path: Path):
    manifest = tmp_path / "manifest.csv"
    splits = tmp_path / "splits.json"
    cache = tmp_path / "cache"

    assert main(["manifest", "--root", str(split_dataset_dir), "--out", str(manifest)]) == 0
    assert manifest.exists()

    # The fixture only has 2 of 8 subgenres, so validation must fail (exit 1).
    assert main(["validate", "--manifest", str(manifest)]) == 1

    assert main(["split", "--manifest", str(manifest), "--out", str(splits)]) == 0
    assert splits.exists()

    assert main(["preprocess", "--manifest", str(manifest), "--cache", str(cache)]) == 0
    assert (cache / "segments.f16").exists()
