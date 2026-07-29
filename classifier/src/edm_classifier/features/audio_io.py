"""Audio loading and short-chunk segmentation.

Loads MP3/AIFF/WAV files into a mono waveform at the configured sample rate and
splits them into fixed-length overlapping segments, matching the Short-chunk CNN
input contract (2-second chunks by default).
"""

from __future__ import annotations

from pathlib import Path

import librosa
import numpy as np

from edm_classifier.config import SUPPORTED_EXTENSIONS, AudioConfig, settings


class UnsupportedAudioFormatError(ValueError):
    """Raised when a file extension is not one of MP3/AIFF/WAV."""


def is_supported(path: str | Path) -> bool:
    """Return True if ``path`` has a supported audio extension."""
    return Path(path).suffix.lower() in SUPPORTED_EXTENSIONS


def load_audio(
    path: str | Path,
    config: AudioConfig | None = None,
) -> np.ndarray:
    """Load an audio file as a 1-D float32 mono waveform.

    Args:
        path: Path to an MP3, AIFF or WAV file.
        config: Audio configuration; defaults to the global settings.

    Returns:
        Mono waveform of shape ``(n_samples,)`` at ``config.sample_rate``.

    Raises:
        UnsupportedAudioFormatError: If the extension is not supported.
        FileNotFoundError: If the file does not exist.
    """
    config = config or settings.audio
    path = Path(path)
    if not is_supported(path):
        raise UnsupportedAudioFormatError(
            f"Unsupported audio format {path.suffix!r}; expected one of {SUPPORTED_EXTENSIONS}."
        )
    if not path.exists():
        raise FileNotFoundError(f"Audio file not found: {path}")

    waveform, _ = librosa.load(path, sr=config.sample_rate, mono=config.mono)
    return waveform.astype(np.float32, copy=False)


def segment_waveform(
    waveform: np.ndarray,
    config: AudioConfig | None = None,
) -> np.ndarray:
    """Split a waveform into fixed-length overlapping segments.

    The final partial segment is zero-padded to a full segment so no audio is
    dropped. A waveform shorter than one segment yields a single padded segment.

    Args:
        waveform: 1-D mono waveform.
        config: Audio configuration; defaults to the global settings.

    Returns:
        Array of shape ``(n_segments, segment_samples)``.
    """
    config = config or settings.audio
    if waveform.ndim != 1:
        raise ValueError(f"Expected a 1-D waveform, got shape {waveform.shape}.")

    seg_len = config.segment_samples
    overlap = min(max(config.segment_overlap, 0.0), 0.95)
    hop = max(1, int(round(seg_len * (1.0 - overlap))))

    # Trim intro/outro, but only if enough audio remains for one full segment.
    intro = config.trim_intro_samples
    outro = config.trim_outro_samples
    if waveform.shape[0] - intro - outro >= seg_len:
        waveform = waveform[intro : waveform.shape[0] - outro]

    n_samples = waveform.shape[0]
    if n_samples <= seg_len:
        padded = np.zeros(seg_len, dtype=waveform.dtype)
        padded[:n_samples] = waveform
        return padded[np.newaxis, :]

    starts = list(range(0, n_samples - seg_len + 1, hop))
    # Ensure the tail of the signal is covered by a final segment.
    if starts[-1] + seg_len < n_samples:
        starts.append(n_samples - seg_len)

    segments = np.stack([waveform[s : s + seg_len] for s in starts], axis=0)
    return segments
