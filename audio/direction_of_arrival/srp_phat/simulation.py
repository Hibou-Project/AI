from __future__ import annotations

import os
import wave

import numpy as np

from .config import MIC_POSITIONS, SAMPLE_RATE, SPEED_OF_SOUND


def simulate_signals(
    source_angle_deg: float,
    fs: int = SAMPLE_RATE,
    duration: float = 1.0,
    snr_db: float = 20.0,
    mic_positions: np.ndarray = MIC_POSITIONS,
) -> np.ndarray:
    """Generate synthetic multi-mic drone-like audio at a known direction."""
    n_samples = int(duration * fs)
    t = np.arange(n_samples) / fs

    drone = (
        0.5 * np.sin(2 * np.pi * 80 * t)
        + 0.3 * np.sin(2 * np.pi * 160 * t)
        + 0.2 * np.sin(2 * np.pi * 240 * t)
        + 0.1 * np.random.randn(n_samples)
    )

    direction = np.array(
        [np.cos(np.radians(source_angle_deg)), np.sin(np.radians(source_angle_deg))]
    )
    noise_power = 10.0 ** (-snr_db / 10.0)
    outputs: list[np.ndarray] = []

    for pos in mic_positions:
        tau = -np.dot(pos, direction) / SPEED_OF_SOUND
        delay = int(round(tau * fs))
        delayed = np.roll(drone, delay)
        noise = np.random.randn(n_samples) * np.sqrt(noise_power)
        outputs.append((delayed + noise).astype(np.float32))

    return np.stack(outputs, axis=0)


def save_simulation_wavs(
    signals: np.ndarray,
    fs: int,
    source_angle_deg: float,
    out_dir: str = "simulation_signals",
) -> list[str]:
    os.makedirs(out_dir, exist_ok=True)
    written: list[str] = []

    for idx, sig in enumerate(signals):
        filename = f"mic{idx + 1}_{int(source_angle_deg):03d}deg.wav"
        path = os.path.join(out_dir, filename)
        pcm = (np.clip(sig, -1.0, 1.0) * 32767).astype(np.int16)
        with wave.open(path, "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(fs)
            wf.writeframes(pcm.tobytes())
        written.append(path)

    return written
