"""Tests for dataset validation against the requirements."""

from __future__ import annotations

from edm_classifier.config import SUBGENRES
from edm_classifier.data.manifest import ManifestEntry
from edm_classifier.data.validate import validate_manifest


def _entry(subgenre: str, label: int, bitrate: float = 320.0, dual: bool = True) -> ManifestEntry:
    return ManifestEntry(
        path=f"/tmp/{label}/track.wav",
        subgenre=subgenre,
        label=label,
        format="WAV",
        sample_rate=22050,
        channels=2,
        duration_seconds=240.0,
        bitrate_kbps=bitrate,
        size_bytes=1_000_000,
        source_1="beatport" if dual else "",
        source_2="soundcloud" if dual else "",
    )


def _full_manifest(per_class: int = 2, **kw) -> list[ManifestEntry]:
    entries = []
    for label, name in enumerate(SUBGENRES):
        for _ in range(per_class):
            entries.append(_entry(name, label, **kw))
    return entries


def test_valid_manifest_passes():
    entries = _full_manifest(per_class=2)
    report = validate_manifest(entries, min_tracks_per_class=2)
    assert report.ok
    assert not report.errors
    assert all(n == 2 for n in report.distribution.values())


def test_missing_class_fails():
    # Only 7 of the 8 subgenres present.
    entries = [e for e in _full_manifest(per_class=2) if e.subgenre != "trance"]
    report = validate_manifest(entries, min_tracks_per_class=2)
    assert not report.ok
    assert any("trance" in e for e in report.errors)


def test_low_bitrate_fails():
    entries = _full_manifest(per_class=2)
    entries[0] = _entry(SUBGENRES[0], 0, bitrate=96.0)
    report = validate_manifest(entries, min_tracks_per_class=2, min_bitrate_kbps=128)
    assert not report.ok
    assert any("kbps" in e for e in report.errors)


def test_dual_source_warning_vs_error():
    entries = _full_manifest(per_class=2, dual=False)
    soft = validate_manifest(entries, min_tracks_per_class=2)
    assert soft.ok  # missing sources is only a warning by default
    assert soft.warnings

    strict = validate_manifest(entries, min_tracks_per_class=2, require_dual_source=True)
    assert not strict.ok


def test_summary_is_readable():
    report = validate_manifest(_full_manifest(per_class=2), min_tracks_per_class=2)
    text = report.summary()
    assert "Dataset validation: OK" in text
    assert "deep house" in text
