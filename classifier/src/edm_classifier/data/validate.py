"""Dataset validation against the project requirements.

Checks a manifest against the spec:
- Req 3.1: a minimum number of tracks per subgenre (config: 100).
- Req 3.3: audio bitrate at or above 128 kbps.
- Supported formats only (MP3/AIFF/WAV).
- Req 3.2 (soft): each track validated by two professional sources.

Returns a structured report instead of raising, so callers can print it, log it
or assert on ``report.ok``.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field

from edm_classifier.config import SUBGENRE_TO_INDEX, SUPPORTED_EXTENSIONS, settings
from edm_classifier.data.manifest import ManifestEntry


@dataclass
class ValidationReport:
    """Outcome of validating a dataset manifest."""

    ok: bool
    distribution: dict[str, int]
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    def summary(self) -> str:
        status = "OK" if self.ok else "FAILED"
        lines = [f"Dataset validation: {status}"]
        for name, n in self.distribution.items():
            lines.append(f"  {name:18s} {n:4d} tracks")
        for e in self.errors:
            lines.append(f"  ERROR: {e}")
        for w in self.warnings:
            lines.append(f"  WARN:  {w}")
        return "\n".join(lines)


def validate_manifest(
    entries: list[ManifestEntry],
    min_tracks_per_class: int | None = None,
    min_bitrate_kbps: int | None = None,
    require_dual_source: bool = False,
) -> ValidationReport:
    """Validate a manifest and return a :class:`ValidationReport`.

    Args:
        entries: The manifest to validate.
        min_tracks_per_class: Required tracks per subgenre (default: config).
        min_bitrate_kbps: Minimum bitrate in kbps (default: config).
        require_dual_source: If True, missing dual-source validation is an error
            rather than a warning.
    """
    cfg = settings.dataset
    min_tracks = min_tracks_per_class if min_tracks_per_class is not None else cfg.tracks_per_class
    min_bitrate = min_bitrate_kbps if min_bitrate_kbps is not None else cfg.min_bitrate_kbps

    errors: list[str] = []
    warnings: list[str] = []

    counts = Counter(e.subgenre for e in entries)
    distribution = {name: counts.get(name, 0) for name in SUBGENRE_TO_INDEX}

    # Req 3.1: minimum tracks per subgenre.
    for name, n in distribution.items():
        if n < min_tracks:
            errors.append(f"{name}: {n} tracks (< required {min_tracks}).")

    # Unknown subgenres (should not happen via the indexer, but guard anyway).
    for name in counts:
        if name not in SUBGENRE_TO_INDEX:
            errors.append(f"Unknown subgenre in manifest: {name!r}.")

    allowed = {ext.lstrip(".").lower() for ext in SUPPORTED_EXTENSIONS}
    n_missing_source = 0
    for e in entries:
        # Req 3.3: bitrate floor.
        if e.bitrate_kbps < min_bitrate:
            errors.append(f"{e.path}: {e.bitrate_kbps:.0f} kbps (< {min_bitrate}).")
        # Supported format.
        if e.format.lower() not in allowed and _ext_of(e.path) not in allowed:
            warnings.append(f"{e.path}: unexpected format {e.format!r}.")
        # Req 3.2: dual professional source.
        if not (e.source_1 and e.source_2):
            n_missing_source += 1

    if n_missing_source:
        msg = f"{n_missing_source} tracks missing dual-source validation (Req 3.2)."
        (errors if require_dual_source else warnings).append(msg)

    return ValidationReport(
        ok=len(errors) == 0,
        distribution=distribution,
        errors=errors,
        warnings=warnings,
    )


def _ext_of(path: str) -> str:
    _, _, ext = path.rpartition(".")
    return ext.lower()
