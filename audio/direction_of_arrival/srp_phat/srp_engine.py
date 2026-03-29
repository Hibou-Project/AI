from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .beam import pair_weight
from .config import (
    BEAM_HALF_WIDTH,
    FREQ_MAX,
    FREQ_MIN,
    FRAME_DURATION,
    FRAME_HOP,
    GAIN_FLOOR,
    MIC_DIRECTIONS,
    MIC_POSITIONS,
    SAMPLE_RATE,
    SPEED_OF_SOUND,
    THETA_RESOLUTION,
)
from .gcc_phat import gcc_phat, tdoa_to_samples
from .peak import refine_peak


@dataclass
class SRPEngine:
    fs: int = SAMPLE_RATE
    mic_positions: np.ndarray = MIC_POSITIONS
    mic_directions: np.ndarray = MIC_DIRECTIONS
    speed_of_sound: float = SPEED_OF_SOUND
    beam_half_width: float = BEAM_HALF_WIDTH
    gain_floor: float = GAIN_FLOOR
    theta_resolution: float = THETA_RESOLUTION
    frame_duration: float = FRAME_DURATION
    frame_hop: float = FRAME_HOP
    freq_min: float = FREQ_MIN
    freq_max: float = FREQ_MAX

    theta_grid: np.ndarray = field(init=False)
    n_fft: int = field(init=False)
    hop: int = field(init=False)
    pairs: list[tuple[int, int]] = field(init=False)
    window: np.ndarray = field(init=False)
    tdoa_table: np.ndarray = field(init=False)
    weight_table: np.ndarray = field(init=False)

    def __post_init__(self) -> None:
        self.theta_grid = np.arange(0.0, 360.0, self.theta_resolution)
        self.n_fft = int(self.frame_duration * self.fs)
        self.hop = int(self.frame_hop * self.fs)
        self.window = np.hanning(self.n_fft)
        self.pairs = [
            (i, j)
            for i in range(len(self.mic_positions))
            for j in range(i + 1, len(self.mic_positions))
        ]
        self._build_steering_tables()

    def _build_steering_tables(self) -> None:
        n_pairs = len(self.pairs)
        n_theta = len(self.theta_grid)
        self.tdoa_table = np.zeros((n_pairs, n_theta), dtype=int)
        self.weight_table = np.zeros((n_pairs, n_theta), dtype=np.float32)

        unit_vectors = np.column_stack(
            [np.cos(np.radians(self.theta_grid)), np.sin(np.radians(self.theta_grid))]
        )

        for pair_idx, (i, j) in enumerate(self.pairs):
            baseline = self.mic_positions[i] - self.mic_positions[j]
            tau = -(unit_vectors @ baseline) / self.speed_of_sound
            self.tdoa_table[pair_idx] = [
                tdoa_to_samples(delay, self.fs, self.n_fft) for delay in tau
            ]
            self.weight_table[pair_idx] = [
                pair_weight(
                    theta=theta,
                    dir_a=float(self.mic_directions[i]),
                    dir_b=float(self.mic_directions[j]),
                    half_beamwidth=self.beam_half_width,
                    gain_floor=self.gain_floor,
                )
                for theta in self.theta_grid
            ]

    def process(self, signals: np.ndarray) -> np.ndarray:
        if signals.ndim != 2:
            raise ValueError("signals must have shape (n_mics, n_samples)")
        if signals.shape[0] != len(self.mic_positions):
            raise ValueError(
                f"Expected {len(self.mic_positions)} mics, got {signals.shape[0]}"
            )

        n_samples = signals.shape[1]
        srp_acc = np.zeros(len(self.theta_grid), dtype=np.float64)
        n_frames = 0
        start = 0

        while start + self.n_fft <= n_samples:
            end = start + self.n_fft
            frames = [
                signals[mic_idx, start:end] * self.window
                for mic_idx in range(len(self.mic_positions))
            ]
            gccs = [
                gcc_phat(
                    frames[i],
                    frames[j],
                    n_fft=self.n_fft,
                    fs=self.fs,
                    freq_min=self.freq_min,
                    freq_max=self.freq_max,
                )
                for i, j in self.pairs
            ]

            srp_frame = np.zeros(len(self.theta_grid), dtype=np.float64)
            for pair_idx, gcc in enumerate(gccs):
                srp_frame += self.weight_table[pair_idx] * gcc[self.tdoa_table[pair_idx]]

            srp_acc += srp_frame
            n_frames += 1
            start += self.hop

        if n_frames == 0:
            return srp_acc
        return srp_acc / n_frames

    def find_doa(self, srp_map: np.ndarray) -> tuple[float, float]:
        doa = refine_peak(srp_map, self.theta_grid)
        peak = float(np.max(np.abs(srp_map)))
        mean = float(np.mean(np.abs(srp_map)))
        confidence = peak / (mean + 1e-12)
        return doa, confidence
