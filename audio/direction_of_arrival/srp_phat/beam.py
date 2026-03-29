from __future__ import annotations

import numpy as np


def parabolic_gain(
    theta_source: float,
    mic_direction: float,
    half_beamwidth: float,
    gain_floor: float,
) -> float:
    """Cosine-tapered directional gain with a floor for sidelobe contribution."""
    off_axis = abs((theta_source - mic_direction + 180.0) % 360.0 - 180.0)
    if off_axis >= half_beamwidth:
        return float(gain_floor)
    main_lobe = np.cos(np.pi * off_axis / (2.0 * half_beamwidth))
    return float(gain_floor + (1.0 - gain_floor) * main_lobe)


def pair_weight(
    theta: float,
    dir_a: float,
    dir_b: float,
    half_beamwidth: float,
    gain_floor: float,
) -> float:
    gain_a = parabolic_gain(theta, dir_a, half_beamwidth, gain_floor)
    gain_b = parabolic_gain(theta, dir_b, half_beamwidth, gain_floor)
    return float(np.sqrt(gain_a * gain_b))
