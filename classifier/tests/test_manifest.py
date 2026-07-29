"""Tests for dataset manifest building, persistence and audio probing."""

from __future__ import annotations

from pathlib import Path

from edm_classifier.config import settings
from edm_classifier.data.manifest import (
    build_manifest,
    load_manifest,
    probe_audio,
    save_manifest,
    to_track_records,
)

SR = settings.audio.sample_rate


def test_probe_audio_reads_metadata(wav_file: Path):
    meta = probe_audio(wav_file)
    assert meta.sample_rate == SR
    assert meta.channels == 1
    assert meta.duration_seconds > 2.9
    assert "WAV" in meta.format.upper()
    assert meta.bitrate_kbps > 128  # PCM is comfortably above the floor
    assert meta.size_bytes > 0


def test_build_manifest_covers_all_tracks(dataset_dir: Path):
    entries = build_manifest(dataset_dir)
    assert len(entries) == 8
    assert {e.subgenre for e in entries} == {"deep house", "trance"}
    for e in entries:
        assert e.sample_rate == SR
        assert e.duration_seconds > 2.9
        assert e.bitrate_kbps > 0


def test_manifest_csv_roundtrip(dataset_dir: Path, tmp_path: Path):
    entries = build_manifest(dataset_dir)
    csv_path = tmp_path / "manifest.csv"
    save_manifest(entries, csv_path)
    assert csv_path.exists()

    loaded = load_manifest(csv_path)
    assert len(loaded) == len(entries)
    assert [e.path for e in loaded] == [e.path for e in entries]
    assert [e.label for e in loaded] == [e.label for e in entries]


def test_to_track_records(dataset_dir: Path):
    entries = build_manifest(dataset_dir)
    records = to_track_records(entries)
    assert len(records) == len(entries)
    assert records[0].subgenre == entries[0].subgenre
    assert records[0].label == entries[0].label
