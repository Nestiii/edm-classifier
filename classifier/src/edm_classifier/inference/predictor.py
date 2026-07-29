"""Track-level inference: mel-segments -> aggregated subgenre + confidence.

A track is split into 2-second segments, each scored by the model, and the
per-segment probabilities are averaged into a single track prediction. The spec
requires a confidence value and the two most probable classes (top-2), both
returned here.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from edm_classifier.config import SUBGENRES, AudioConfig, FeatureConfig, settings
from edm_classifier.features.pipeline import track_to_model_input
from edm_classifier.models.short_chunk_cnn import ShortChunkCNNRes, build_model


@dataclass(frozen=True)
class Prediction:
    """Track-level prediction result."""

    subgenre: str
    confidence: float
    top2: list[tuple[str, float]]  # [(label, prob), (label, prob)] sorted desc
    probabilities: dict[str, float]  # full distribution over subgenres

    @property
    def label_index(self) -> int:
        return SUBGENRES.index(self.subgenre)


class Predictor:
    """Wraps a trained model to classify audio tracks."""

    def __init__(
        self,
        model: ShortChunkCNNRes,
        device: str | torch.device = "cpu",
        audio: AudioConfig | None = None,
        feat: FeatureConfig | None = None,
    ) -> None:
        self.device = torch.device(device)
        self.model = model.to(self.device).eval()
        self.audio = audio or settings.audio
        self.feat = feat or settings.features

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint_path: str | Path,
        device: str | torch.device = "cpu",
        **model_kwargs: int,
    ) -> Predictor:
        """Load a serialized state_dict into a fresh model."""
        model = build_model(**model_kwargs)
        state = torch.load(checkpoint_path, map_location="cpu")
        # Accept either a bare state_dict or a {'model': state_dict} wrapper.
        if isinstance(state, dict) and "model" in state:
            state = state["model"]
        model.load_state_dict(state)
        return cls(model, device=device)

    @torch.no_grad()
    def predict_segments(self, segments: np.ndarray) -> np.ndarray:
        """Return ``(n_segments, n_class)`` softmax probabilities for segments."""
        x = torch.from_numpy(np.ascontiguousarray(segments)).float().to(self.device)
        logits = self.model(x)
        return torch.softmax(logits, dim=1).cpu().numpy()

    @torch.no_grad()
    def predict_track(self, path: str | Path) -> Prediction:
        """Classify a whole audio track and return an aggregated prediction."""
        segments = track_to_model_input(path, self.audio, self.feat)
        seg_probs = self.predict_segments(segments)
        track_probs = seg_probs.mean(axis=0)
        return self._to_prediction(track_probs)

    @staticmethod
    def _to_prediction(probs: np.ndarray) -> Prediction:
        probs = np.asarray(probs, dtype=np.float64)
        order = np.argsort(probs)[::-1]
        best = int(order[0])
        top2 = [(SUBGENRES[int(i)], float(probs[int(i)])) for i in order[:2]]
        distribution = {SUBGENRES[i]: float(probs[i]) for i in range(len(SUBGENRES))}
        return Prediction(
            subgenre=SUBGENRES[best],
            confidence=float(probs[best]),
            top2=top2,
            probabilities=distribution,
        )
