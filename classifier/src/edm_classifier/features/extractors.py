"""Spectral and temporal feature extractors built on Librosa.

Implements every feature required by the spec: mel-spectrograms, MFCCs,
tempograms, spectral centroid, spectral rolloff, zero-crossing rate and chroma.
Each extractor takes a mono waveform and returns a 2-D ``(n_features, n_frames)``
array so features can be stacked or summarised uniformly downstream.
"""

from __future__ import annotations

from dataclasses import dataclass

import librosa
import numpy as np

from edm_classifier.config import AudioConfig, FeatureConfig, settings


def _fmax(audio: AudioConfig, feat: FeatureConfig) -> float:
    return feat.fmax if feat.fmax is not None else audio.sample_rate / 2


def mel_spectrogram(
    waveform: np.ndarray,
    audio: AudioConfig | None = None,
    feat: FeatureConfig | None = None,
    *,
    to_db: bool = True,
) -> np.ndarray:
    """Compute a (log-)mel-spectrogram of shape ``(n_mels, n_frames)``.

    This is the primary input representation for the Short-chunk CNN.
    """
    audio = audio or settings.audio
    feat = feat or settings.features
    mel = librosa.feature.melspectrogram(
        y=waveform,
        sr=audio.sample_rate,
        n_fft=feat.n_fft,
        hop_length=feat.hop_length,
        win_length=feat.win_length,
        n_mels=feat.n_mels,
        fmin=feat.fmin,
        fmax=_fmax(audio, feat),
        power=2.0,
    )
    if to_db:
        mel = librosa.power_to_db(mel, ref=np.max)
    return mel.astype(np.float32, copy=False)


def mfcc(
    waveform: np.ndarray,
    audio: AudioConfig | None = None,
    feat: FeatureConfig | None = None,
) -> np.ndarray:
    """Compute MFCCs of shape ``(n_mfcc, n_frames)``."""
    audio = audio or settings.audio
    feat = feat or settings.features
    out = librosa.feature.mfcc(
        y=waveform,
        sr=audio.sample_rate,
        n_mfcc=feat.n_mfcc,
        n_fft=feat.n_fft,
        hop_length=feat.hop_length,
    )
    return out.astype(np.float32, copy=False)


def tempogram(
    waveform: np.ndarray,
    audio: AudioConfig | None = None,
    feat: FeatureConfig | None = None,
) -> np.ndarray:
    """Compute a tempogram of shape ``(win_length, n_frames)``.

    Captures local rhythmic (tempo) structure, useful for distinguishing the
    four-on-the-floor patterns that separate EDM subgenres.
    """
    audio = audio or settings.audio
    feat = feat or settings.features
    onset_env = librosa.onset.onset_strength(
        y=waveform, sr=audio.sample_rate, hop_length=feat.hop_length
    )
    out = librosa.feature.tempogram(
        onset_envelope=onset_env, sr=audio.sample_rate, hop_length=feat.hop_length
    )
    return out.astype(np.float32, copy=False)


def spectral_centroid(
    waveform: np.ndarray,
    audio: AudioConfig | None = None,
    feat: FeatureConfig | None = None,
) -> np.ndarray:
    """Compute the spectral centroid of shape ``(1, n_frames)``."""
    audio = audio or settings.audio
    feat = feat or settings.features
    out = librosa.feature.spectral_centroid(
        y=waveform, sr=audio.sample_rate, n_fft=feat.n_fft, hop_length=feat.hop_length
    )
    return out.astype(np.float32, copy=False)


def spectral_rolloff(
    waveform: np.ndarray,
    audio: AudioConfig | None = None,
    feat: FeatureConfig | None = None,
) -> np.ndarray:
    """Compute the spectral rolloff of shape ``(1, n_frames)``."""
    audio = audio or settings.audio
    feat = feat or settings.features
    out = librosa.feature.spectral_rolloff(
        y=waveform, sr=audio.sample_rate, n_fft=feat.n_fft, hop_length=feat.hop_length
    )
    return out.astype(np.float32, copy=False)


def zero_crossing_rate(
    waveform: np.ndarray,
    feat: FeatureConfig | None = None,
) -> np.ndarray:
    """Compute the zero-crossing rate of shape ``(1, n_frames)``."""
    feat = feat or settings.features
    out = librosa.feature.zero_crossing_rate(
        y=waveform, frame_length=feat.n_fft, hop_length=feat.hop_length
    )
    return out.astype(np.float32, copy=False)


def chroma(
    waveform: np.ndarray,
    audio: AudioConfig | None = None,
    feat: FeatureConfig | None = None,
) -> np.ndarray:
    """Compute a chroma (STFT) feature of shape ``(n_chroma, n_frames)``."""
    audio = audio or settings.audio
    feat = feat or settings.features
    out = librosa.feature.chroma_stft(
        y=waveform,
        sr=audio.sample_rate,
        n_fft=feat.n_fft,
        hop_length=feat.hop_length,
        n_chroma=feat.n_chroma,
    )
    return out.astype(np.float32, copy=False)


# Registry mapping each feature name to its extractor. Extractors that don't
# take an ``audio`` argument are wrapped so every entry has a uniform signature.
@dataclass(frozen=True)
class FeatureSet:
    """Bundle of every extracted feature for a single waveform segment."""

    mel_spectrogram: np.ndarray
    mfcc: np.ndarray
    tempogram: np.ndarray
    spectral_centroid: np.ndarray
    spectral_rolloff: np.ndarray
    zero_crossing_rate: np.ndarray
    chroma: np.ndarray

    def as_dict(self) -> dict[str, np.ndarray]:
        return {
            "mel_spectrogram": self.mel_spectrogram,
            "mfcc": self.mfcc,
            "tempogram": self.tempogram,
            "spectral_centroid": self.spectral_centroid,
            "spectral_rolloff": self.spectral_rolloff,
            "zero_crossing_rate": self.zero_crossing_rate,
            "chroma": self.chroma,
        }


def extract_all(
    waveform: np.ndarray,
    audio: AudioConfig | None = None,
    feat: FeatureConfig | None = None,
) -> FeatureSet:
    """Extract the full set of required features for one waveform."""
    audio = audio or settings.audio
    feat = feat or settings.features
    return FeatureSet(
        mel_spectrogram=mel_spectrogram(waveform, audio, feat),
        mfcc=mfcc(waveform, audio, feat),
        tempogram=tempogram(waveform, audio, feat),
        spectral_centroid=spectral_centroid(waveform, audio, feat),
        spectral_rolloff=spectral_rolloff(waveform, audio, feat),
        zero_crossing_rate=zero_crossing_rate(waveform, feat),
        chroma=chroma(waveform, audio, feat),
    )


def summary_vector(features: FeatureSet) -> np.ndarray:
    """Reduce a FeatureSet to a fixed-length vector of per-feature mean+std.

    Useful for classical baselines and for logging/inspecting features without
    the full time axis.
    """
    parts: list[np.ndarray] = []
    for arr in features.as_dict().values():
        parts.append(arr.mean(axis=1))
        parts.append(arr.std(axis=1))
    return np.concatenate(parts).astype(np.float32, copy=False)
