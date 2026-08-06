"""Lazy loader for the inference model behind the API.

Keeps a single :class:`~edm_classifier.inference.predictor.Predictor` alive for
the process, loaded on first use from the configured checkpoint. Tests can inject
a fake predictor instead of loading real weights.
"""

from __future__ import annotations

from pathlib import Path

from edm_classifier.config import settings


def resolve_device(device: str) -> str:
    """Resolve ``"auto"`` to the best available torch device string."""
    if device != "auto":
        return device
    import torch

    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


class ModelService:
    """Holds the predictor, loading it lazily from a checkpoint."""

    def __init__(self, predictor=None) -> None:
        self._predictor = predictor  # may be injected (tests) or lazily loaded
        self._device: str | None = None

    @property
    def is_loaded(self) -> bool:
        return self._predictor is not None

    @property
    def device(self) -> str | None:
        return self._device

    def set_predictor(self, predictor, device: str = "injected") -> None:
        """Inject a ready predictor (used in tests)."""
        self._predictor = predictor
        self._device = device

    def load(self, model_path: str | Path | None = None, device: str | None = None) -> None:
        """Load the predictor from a checkpoint. No-op if already loaded."""
        if self._predictor is not None:
            return
        path = model_path or settings.api.model_path
        if not path:
            raise RuntimeError(
                "No model checkpoint configured. Set EDM_API_MODEL_PATH or call load(path)."
            )
        if not Path(path).exists():
            raise FileNotFoundError(f"Model checkpoint not found: {path}")

        # Imported lazily so importing the API package doesn't require torch.
        from edm_classifier.inference.predictor import Predictor

        resolved = resolve_device(device or settings.api.device)
        self._predictor = Predictor.from_checkpoint(path, device=resolved)
        self._device = resolved

    def predictor(self):
        """Return the loaded predictor, loading it on demand."""
        if self._predictor is None:
            self.load()
        return self._predictor
