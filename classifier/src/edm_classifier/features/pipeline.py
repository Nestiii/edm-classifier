"""High-level feature pipeline: from an audio file to model-ready tensors.

Ties together audio loading, short-chunk segmentation and mel-spectrogram
extraction, producing the batched input the Short-chunk CNN consumes.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from edm_classifier.config import AudioConfig, FeatureConfig, settings
from edm_classifier.features.audio_io import load_audio, segment_waveform
from edm_classifier.features.extractors import mel_spectrogram


def track_to_segments(
    path: str | Path,
    audio: AudioConfig | None = None,
) -> np.ndarray:
    """Load a track and split it into fixed-length segments.

    Returns an array of shape ``(n_segments, segment_samples)``.
    """
    audio = audio or settings.audio
    waveform = load_audio(path, audio)
    return segment_waveform(waveform, audio)


def track_to_model_input(
    path: str | Path,
    audio: AudioConfig | None = None,
    feat: FeatureConfig | None = None,
) -> np.ndarray:
    """Turn a track into batched mel-spectrograms for the CNN.

    Returns an array of shape ``(n_segments, 1, n_mels, n_frames)`` — one
    single-channel mel-spectrogram image per 2-second segment.
    """
    audio = audio or settings.audio
    feat = feat or settings.features
    segments = track_to_segments(path, audio)
    mels = np.stack([mel_spectrogram(seg, audio, feat) for seg in segments], axis=0)
    # Add the channel dimension expected by a 2-D CNN.
    return mels[:, np.newaxis, :, :]
