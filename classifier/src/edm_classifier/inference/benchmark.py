"""Per-track processing-time benchmark (Req 1.5: < 5 s per track).

Measures the wall-clock time of a full track classification — load audio, extract
features, run the model, aggregate — which is what the requirement bounds.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path

from edm_classifier.config import SUPPORTED_EXTENSIONS
from edm_classifier.inference.predictor import Predictor


@dataclass(frozen=True)
class BenchmarkResult:
    """Timing summary over a set of tracks."""

    n_tracks: int
    times: list[float]
    device: str

    @property
    def mean(self) -> float:
        return sum(self.times) / len(self.times)

    @property
    def maximum(self) -> float:
        return max(self.times)

    @property
    def minimum(self) -> float:
        return min(self.times)

    def meets_target(self, target_seconds: float = 5.0) -> bool:
        """True if every track processed under the target time."""
        return self.maximum <= target_seconds

    def summary(self, target_seconds: float = 5.0) -> dict[str, object]:
        return {
            "n_tracks": self.n_tracks,
            "device": self.device,
            "mean_seconds": round(self.mean, 3),
            "max_seconds": round(self.maximum, 3),
            "min_seconds": round(self.minimum, 3),
            "target_seconds": target_seconds,
            "meets_target": self.meets_target(target_seconds),
        }


def benchmark_tracks(
    predictor: Predictor,
    paths: list[str | Path],
    warmup: int = 1,
) -> BenchmarkResult:
    """Time ``predictor.predict_track`` over each path.

    Args:
        predictor: A loaded predictor.
        paths: Audio files to time.
        warmup: Number of leading, untimed runs to absorb lazy init / JIT / caches.

    Returns:
        A :class:`BenchmarkResult`.
    """
    if not paths:
        raise ValueError("No tracks to benchmark.")

    # Warm up on the first track(s) so timings reflect steady state.
    for path in paths[: max(0, warmup)]:
        predictor.predict_track(path)

    times: list[float] = []
    for path in paths:
        start = time.perf_counter()
        predictor.predict_track(path)
        times.append(time.perf_counter() - start)

    return BenchmarkResult(n_tracks=len(paths), times=times, device=str(predictor.device))


def find_audio(directory: str | Path) -> list[Path]:
    """List supported audio files directly in ``directory`` (sorted)."""
    directory = Path(directory)
    return sorted(
        p for p in directory.glob("*") if p.is_file() and p.suffix.lower() in SUPPORTED_EXTENSIONS
    )
