from __future__ import annotations

from . import config
from .audio import load_signals
from .peak import find_top_k_peaks
from .srp_engine import SRPEngine


def estimate_doa(
    signals,
    fs: int = config.SAMPLE_RATE,
    top_k: int = 3,
):
    engine = SRPEngine(fs=fs)
    srp_map = engine.process(signals)
    doa, confidence = engine.find_doa(srp_map)
    peaks = find_top_k_peaks(srp_map, engine.theta_grid, k=top_k)
    return doa, confidence, peaks, srp_map, engine.theta_grid


__all__ = ["SRPEngine", "config", "estimate_doa", "load_signals"]
