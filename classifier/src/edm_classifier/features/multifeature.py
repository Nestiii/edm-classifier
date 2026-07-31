"""Aligned multi-feature extraction for the v3 fusion model.

Produces, per 2-second-hop 4-second segment, three time-aligned representations:
mel-spectrogram, Fourier tempogram and autocorrelation tempogram. Tempograms need
a longer window than a 4 s chunk to estimate tempo stably (the ~8.9 s tempogram
window exceeds a 4 s segment), so all three features are computed **over the whole
(intro/outro-trimmed) track** and then sliced by frame range per segment — the
alignment approach of Hsu et al. 2021 (arXiv:2110.08862).
"""

from __future__ import annotations

from pathlib import Path

import librosa
import numpy as np

from edm_classifier.config import AudioConfig, FeatureConfig, settings
from edm_classifier.features.audio_io import load_audio
from edm_classifier.features.extractors import mel_spectrogram


def frames_per_segment(audio: AudioConfig, feat: FeatureConfig) -> int:
    """Number of feature frames spanned by one segment (matches mel framing)."""
    return 1 + audio.segment_samples // feat.hop_length


def _segment_starts(n_frames: int, seg_frames: int, hop_frames: int) -> list[int]:
    """Frame start indices covering the track, with a tail-aligned final segment."""
    if n_frames <= seg_frames:
        return [0]
    starts = list(range(0, n_frames - seg_frames + 1, hop_frames))
    if starts[-1] + seg_frames < n_frames:
        starts.append(n_frames - seg_frames)
    return starts


def _slice_pad(feature: np.ndarray, start: int, seg_frames: int) -> np.ndarray:
    """Slice ``seg_frames`` columns from a ``(bins, n_frames)`` feature, zero-padded."""
    seg = feature[:, start : start + seg_frames]
    if seg.shape[1] < seg_frames:
        pad = np.zeros((feature.shape[0], seg_frames - seg.shape[1]), dtype=feature.dtype)
        seg = np.concatenate([seg, pad], axis=1)
    return seg


def _trim(waveform: np.ndarray, audio: AudioConfig) -> np.ndarray:
    intro, outro = audio.trim_intro_samples, audio.trim_outro_samples
    if waveform.shape[0] - intro - outro >= audio.segment_samples:
        return waveform[intro : waveform.shape[0] - outro]
    return waveform


def track_to_multifeature(
    path: str | Path,
    audio: AudioConfig | None = None,
    feat: FeatureConfig | None = None,
) -> dict[str, np.ndarray]:
    """Extract aligned mel / Fourier-tempogram / autocorr-tempogram per segment.

    Returns a dict with keys ``"mel"``, ``"fourier"``, ``"autocorr"``; each value
    has shape ``(n_segments, 1, bins, seg_frames)`` (float32).
    """
    audio = audio or settings.audio
    feat = feat or settings.features

    waveform = _trim(load_audio(path, audio), audio)
    onset = librosa.onset.onset_strength(
        y=waveform, sr=audio.sample_rate, hop_length=feat.hop_length
    )
    mel = mel_spectrogram(waveform, audio, feat)
    fourier = np.abs(
        librosa.feature.fourier_tempogram(
            onset_envelope=onset,
            sr=audio.sample_rate,
            hop_length=feat.hop_length,
            win_length=feat.tempogram_win_length,
        )
    ).astype(np.float32)
    autocorr = librosa.feature.tempogram(
        onset_envelope=onset,
        sr=audio.sample_rate,
        hop_length=feat.hop_length,
        win_length=feat.tempogram_win_length,
    ).astype(np.float32)

    # All three share the same hop, so frame counts match up to rounding.
    n = min(mel.shape[1], fourier.shape[1], autocorr.shape[1])
    mel, fourier, autocorr = mel[:, :n], fourier[:, :n], autocorr[:, :n]

    seg_frames = frames_per_segment(audio, feat)
    overlap = min(max(audio.segment_overlap, 0.0), 0.95)
    hop_frames = max(1, int(round(seg_frames * (1.0 - overlap))))
    starts = _segment_starts(n, seg_frames, hop_frames)

    def stack(feature: np.ndarray) -> np.ndarray:
        segs = np.stack([_slice_pad(feature, s, seg_frames) for s in starts], axis=0)
        return segs[:, np.newaxis, :, :].astype(np.float32, copy=False)

    return {"mel": stack(mel), "fourier": stack(fourier), "autocorr": stack(autocorr)}
