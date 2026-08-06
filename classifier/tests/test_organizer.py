"""Tests for the per-subgenre file organizer."""

from __future__ import annotations

from pathlib import Path

from edm_classifier.api.organizer import organize_file, unique_destination
from edm_classifier.config import dirname_for


def _touch(path: Path) -> Path:
    path.write_bytes(b"audio")
    return path


def test_move_creates_subgenre_folder(tmp_path: Path):
    src = _touch(tmp_path / "track.wav")
    dest = organize_file(src, tmp_path, "deep house", move=True)
    assert dest == tmp_path / dirname_for("deep house") / "track.wav"
    assert dest.exists()
    assert not src.exists()  # moved


def test_copy_keeps_original(tmp_path: Path):
    src = _touch(tmp_path / "track.wav")
    dest = organize_file(src, tmp_path, "trance", move=False)
    assert dest.exists()
    assert src.exists()  # copied, original remains


def test_name_collision_is_suffixed(tmp_path: Path):
    dest_dir = tmp_path / dirname_for("tech house")
    dest_dir.mkdir(parents=True)
    _touch(dest_dir / "track.wav")  # pre-existing collision

    src = _touch(tmp_path / "track.wav")
    dest = organize_file(src, tmp_path, "tech house", move=True)
    assert dest.name == "track (1).wav"
    assert dest.exists()


def test_unique_destination(tmp_path: Path):
    assert unique_destination(tmp_path, "a.wav") == tmp_path / "a.wav"
    _touch(tmp_path / "a.wav")
    assert unique_destination(tmp_path, "a.wav") == tmp_path / "a (1).wav"
