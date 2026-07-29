"""PyTorch dataset producing 2-second mel-spectrogram segments per track.

Each item is a whole track expanded into its segments, kept together so that
inference can aggregate segment predictions back to a single track-level result.
``collate_segments`` flattens a batch of tracks into a flat batch of segments for
training, while remembering which segments belong to which track.
"""

from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import Dataset

from edm_classifier.config import AudioConfig, FeatureConfig, settings
from edm_classifier.data.dataset import TrackRecord
from edm_classifier.features.pipeline import track_to_model_input


class TrackSegmentDataset(Dataset):
    """Maps a list of tracks to (segments, label) tensors.

    Item ``i`` returns:
        segments: float tensor ``(n_segments, 1, n_mels, n_frames)``
        label: long scalar tensor
    """

    def __init__(
        self,
        records: list[TrackRecord],
        audio: AudioConfig | None = None,
        feat: FeatureConfig | None = None,
    ) -> None:
        self.records = records
        self.audio = audio or settings.audio
        self.feat = feat or settings.features

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        record = self.records[idx]
        segments = track_to_model_input(record.path, self.audio, self.feat)
        x = torch.from_numpy(np.ascontiguousarray(segments)).float()
        y = torch.tensor(record.label, dtype=torch.long)
        return x, y


def collate_segments(
    batch: list[tuple[torch.Tensor, torch.Tensor]],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Flatten a batch of per-track segments into a flat segment batch.

    Returns:
        segments: ``(total_segments, 1, n_mels, n_frames)``
        labels: ``(total_segments,)`` — each segment carries its track's label
        track_ids: ``(total_segments,)`` — index of the source track within the
            batch, so predictions can be aggregated per track.
    """
    seg_tensors: list[torch.Tensor] = []
    labels: list[torch.Tensor] = []
    track_ids: list[torch.Tensor] = []
    for track_idx, (segments, label) in enumerate(batch):
        n = segments.shape[0]
        seg_tensors.append(segments)
        labels.append(label.repeat(n))
        track_ids.append(torch.full((n,), track_idx, dtype=torch.long))
    return (
        torch.cat(seg_tensors, dim=0),
        torch.cat(labels, dim=0),
        torch.cat(track_ids, dim=0),
    )
