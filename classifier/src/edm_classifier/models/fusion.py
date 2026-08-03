"""Late-fusion model: mel-spectrogram + Fourier/autocorrelation tempograms.

Reproduces the spirit of Hsu et al. 2021 (arXiv:2110.08862): the mel-spectrogram
is processed by the Short-chunk CNN + ResNet backbone, each tempogram by a branch
of multi-scale 1-D convolutions over time, and the three pooled representations
are concatenated before the classification head (late fusion).
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from edm_classifier.config import NUM_CLASSES, settings
from edm_classifier.models.short_chunk_cnn import ResBlock2d


def _fourier_bins() -> int:
    return settings.features.tempogram_win_length // 2 + 1


def _autocorr_bins() -> int:
    return settings.features.tempogram_win_length


class MelBranch(nn.Module):
    """Short-chunk CNN + ResNet feature extractor (no classification head)."""

    def __init__(self, n_channels: int = 64) -> None:
        super().__init__()
        c = n_channels
        self.bn_init = nn.BatchNorm2d(1)
        self.blocks = nn.Sequential(
            ResBlock2d(1, c, stride=2),
            ResBlock2d(c, c, stride=2),
            ResBlock2d(c, c * 2, stride=2),
            ResBlock2d(c * 2, c * 2, stride=2),
            ResBlock2d(c * 2, c * 2, stride=2),
            ResBlock2d(c * 2, c * 4, stride=2),
            ResBlock2d(c * 4, c * 4, stride=2),
        )
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.out_dim = c * 4

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.bn_init(x)
        x = self.blocks(x)
        return self.global_pool(x).flatten(1)


class TempoBranch(nn.Module):
    """Multi-scale 1-D conv branch over a tempogram's time axis.

    Input ``(batch, 1, n_bins, n_frames)`` -> pooled feature ``(batch, 4*out)``.
    """

    def __init__(self, n_bins: int, out_channels: int = 64) -> None:
        super().__init__()
        # Normalize each tempo bin (stand-in for the paper's z-score) before the
        # convs: raw Fourier/autocorr magnitudes are on a very different scale
        # than the (dB) mel, and feeding them unnormalized cripples this branch.
        self.bn_in = nn.BatchNorm1d(n_bins)
        # Four parallel 1-D convs with different kernels/strides (Pons et al.).
        specs = [(3, 2), (3, 3), (5, 3), (5, 5)]
        self.convs = nn.ModuleList(
            [nn.Conv1d(n_bins, out_channels, k, stride=s, padding=k // 2) for k, s in specs]
        )
        self.bn = nn.BatchNorm1d(out_channels * len(specs))
        self.out_dim = out_channels * len(specs)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.bn_in(x.squeeze(1))  # (batch, n_bins, n_frames), per-bin normalized
        feats = [F.adaptive_avg_pool1d(F.relu(conv(x)), 1).flatten(1) for conv in self.convs]
        return self.bn(torch.cat(feats, dim=1))


class FusionModel(nn.Module):
    """Late-fusion classifier over mel + two tempograms."""

    def __init__(
        self,
        n_class: int = NUM_CLASSES,
        n_channels: int = 64,
        tempo_channels: int = 64,
        dropout: float = 0.5,
        fourier_bins: int | None = None,
        autocorr_bins: int | None = None,
    ) -> None:
        super().__init__()
        self.mel = MelBranch(n_channels)
        self.fourier = TempoBranch(fourier_bins or _fourier_bins(), tempo_channels)
        self.autocorr = TempoBranch(autocorr_bins or _autocorr_bins(), tempo_channels)

        feat_dim = self.mel.out_dim + self.fourier.out_dim + self.autocorr.out_dim
        hidden = n_channels * 4
        self.head = nn.Sequential(
            nn.Linear(feat_dim, hidden),
            nn.BatchNorm1d(hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, n_class),
        )

    def forward(
        self, mel: torch.Tensor, fourier: torch.Tensor, autocorr: torch.Tensor
    ) -> torch.Tensor:
        m = self.mel(mel)
        f = self.fourier(fourier)
        a = self.autocorr(autocorr)
        return self.head(torch.cat([m, f, a], dim=1))


def build_fusion_model(
    n_class: int = NUM_CLASSES,
    n_channels: int = 64,
    tempo_channels: int = 64,
    dropout: float = 0.5,
) -> FusionModel:
    """Factory returning a configured fusion model."""
    return FusionModel(
        n_class=n_class,
        n_channels=n_channels,
        tempo_channels=tempo_channels,
        dropout=dropout,
    )
