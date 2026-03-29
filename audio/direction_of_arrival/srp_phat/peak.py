from __future__ import annotations

import numpy as np


def refine_peak(srp_map: np.ndarray, theta_grid: np.ndarray) -> float:
    """Parabolic interpolation around argmax for sub-grid DOA precision."""
    idx = int(np.argmax(srp_map))
    n = len(srp_map)

    y0 = srp_map[(idx - 1) % n]
    y1 = srp_map[idx]
    y2 = srp_map[(idx + 1) % n]
    denom = 2.0 * (2.0 * y1 - y0 - y2)
    if abs(denom) < 1e-12:
        return float(theta_grid[idx])

    delta = (y0 - y2) / denom
    step = theta_grid[1] - theta_grid[0]
    return float((theta_grid[idx] + delta * step) % 360.0)


def find_top_k_peaks(
    srp_map: np.ndarray,
    theta_grid: np.ndarray,
    k: int = 3,
    min_separation_deg: float = 30.0,
) -> list[tuple[float, float]]:
    map_copy = srp_map.copy()
    step = theta_grid[1] - theta_grid[0]
    sep_bins = max(1, int(round(min_separation_deg / step)))
    peaks: list[tuple[float, float]] = []

    for _ in range(k):
        idx = int(np.argmax(map_copy))
        peaks.append((float(theta_grid[idx]), float(map_copy[idx])))
        for offset in range(-sep_bins, sep_bins + 1):
            map_copy[(idx + offset) % len(map_copy)] = 0.0

    return peaks
