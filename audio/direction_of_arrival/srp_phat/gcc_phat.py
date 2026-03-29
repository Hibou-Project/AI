from __future__ import annotations

import numpy as np


def gcc_phat(
    x1: np.ndarray,
    x2: np.ndarray,
    n_fft: int,
    fs: int,
    freq_min: float,
    freq_max: float,
) -> np.ndarray:
    """Return GCC-PHAT time-domain correlation (length n_fft)."""
    x1_fft = np.fft.rfft(x1, n=n_fft)
    x2_fft = np.fft.rfft(x2, n=n_fft)

    cross = x1_fft * np.conj(x2_fft)
    cross /= np.abs(cross) + 1e-8

    freqs = np.fft.rfftfreq(n_fft, d=1.0 / fs)
    cross[(freqs < freq_min) | (freqs > freq_max)] = 0.0
    return np.fft.irfft(cross, n=n_fft).real


def tdoa_to_samples(tau_seconds: float, fs: int, n_fft: int) -> int:
    return int(round(tau_seconds * fs)) % n_fft
