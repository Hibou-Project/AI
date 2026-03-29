from __future__ import annotations

from math import gcd
import wave

import numpy as np
from scipy.signal import resample_poly

from .config import SAMPLE_RATE


def load_wav(path: str) -> tuple[np.ndarray, int]:
    """Load mono samples from a 16-bit PCM WAV file."""
    with wave.open(path, "rb") as wf:
        n_channels = wf.getnchannels()
        sample_width = wf.getsampwidth()
        fs = wf.getframerate()
        raw = wf.readframes(wf.getnframes())

    if sample_width != 2:
        raise ValueError(
            f"{path}: only 16-bit PCM WAV is supported in this modular version."
        )

    samples = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
    if n_channels > 1:
        samples = samples[::n_channels]
    return samples, fs


def resample_to(signal: np.ndarray, src_fs: int, dst_fs: int) -> np.ndarray:
    if src_fs == dst_fs:
        return signal.astype(np.float32, copy=False)
    g = gcd(src_fs, dst_fs)
    return resample_poly(signal, dst_fs // g, src_fs // g).astype(np.float32)


def load_signals(
    paths: list[str], target_fs: int = SAMPLE_RATE
) -> tuple[np.ndarray, int]:
    """Load, resample to target_fs, and align all channels to shortest length."""
    if not paths:
        raise ValueError("No input WAV files were provided.")

    loaded = [load_wav(path) for path in paths]
    resampled = [resample_to(sig, fs, target_fs) for sig, fs in loaded]
    min_len = min(len(sig) for sig in resampled)
    aligned = [sig[:min_len] for sig in resampled]
    return np.stack(aligned, axis=0), target_fs
