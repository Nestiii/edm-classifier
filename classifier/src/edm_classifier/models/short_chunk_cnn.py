"""Short-chunk CNN with ResNet residual blocks (PyTorch).

Architecture based on Won et al., "Evaluation of CNN-based Automatic Music
Tagging Models" (the ``ShortChunkCNN_Res`` variant), adapted for 8-way EDM
subgenre classification. Input is a single-channel mel-spectrogram of a
~2-second chunk; the network stacks 7 residual convolutional blocks followed by
global pooling and a small MLP head, as required by the spec.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from edm_classifier.config import NUM_CLASSES, settings


class ResBlock2d(nn.Module):
    """A 2-D residual block: two conv+BN layers with a projected skip path."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 2,
    ) -> None:
        super().__init__()
        padding = kernel_size // 2
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size, 1, padding)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()

        # Project the identity path when shape changes (stride or channel count).
        self.needs_projection = stride != 1 or in_channels != out_channels
        if self.needs_projection:
            self.proj_conv = nn.Conv2d(
                in_channels, out_channels, kernel_size, stride, padding
            )
            self.proj_bn = nn.BatchNorm2d(out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        identity = self.proj_bn(self.proj_conv(x)) if self.needs_projection else x
        return self.relu(out + identity)


class ShortChunkCNNRes(nn.Module):
    """Short-chunk CNN with residual blocks for EDM subgenre classification."""

    def __init__(
        self,
        n_class: int = NUM_CLASSES,
        n_channels: int = 128,
        dropout: float = 0.5,
    ) -> None:
        super().__init__()
        c = n_channels
        self.bn_init = nn.BatchNorm2d(1)

        # 7 residual blocks; channels widen 128 -> 256 -> 512 in stages.
        self.blocks = nn.Sequential(
            ResBlock2d(1, c, stride=2),
            ResBlock2d(c, c, stride=2),
            ResBlock2d(c, c * 2, stride=2),
            ResBlock2d(c * 2, c * 2, stride=2),
            ResBlock2d(c * 2, c * 2, stride=2),
            ResBlock2d(c * 2, c * 4, stride=2),
            ResBlock2d(c * 4, c * 4, stride=2),
        )

        # Global pooling makes the head independent of the exact input length.
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))

        self.head = nn.Sequential(
            nn.Linear(c * 4, c * 4),
            nn.BatchNorm1d(c * 4),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(c * 4, n_class),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return raw logits of shape ``(batch, n_class)``.

        Args:
            x: Mel-spectrogram batch ``(batch, 1, n_mels, n_frames)``.
        """
        if x.dim() != 4 or x.shape[1] != 1:
            raise ValueError(
                f"Expected input (batch, 1, n_mels, n_frames), got {tuple(x.shape)}."
            )
        x = self.bn_init(x)
        x = self.blocks(x)
        x = self.global_pool(x).flatten(1)
        return self.head(x)


def build_model(
    n_class: int = NUM_CLASSES,
    n_channels: int = 128,
    dropout: float = 0.5,
) -> ShortChunkCNNRes:
    """Factory returning a configured model instance."""
    return ShortChunkCNNRes(n_class=n_class, n_channels=n_channels, dropout=dropout)


# Convenience constant: expected number of mel bands the model is trained on.
N_MELS = settings.features.n_mels
