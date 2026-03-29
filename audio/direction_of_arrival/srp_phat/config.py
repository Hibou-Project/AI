from __future__ import annotations

import numpy as np

# Audio / physics
SAMPLE_RATE = 16000
SPEED_OF_SOUND = 343.0

# 2-mic array geometry
MIC_DISTANCE = 0.50  # meters
_MIC_RADIUS = MIC_DISTANCE / np.sqrt(2.0)
MIC_POSITIONS = np.array(
    [
        [_MIC_RADIUS * np.cos(np.radians(0.0)), _MIC_RADIUS * np.sin(np.radians(0.0))],
        [_MIC_RADIUS * np.cos(np.radians(45.0)), _MIC_RADIUS * np.sin(np.radians(45.0))],
    ],
    dtype=np.float32,
)
MIC_DIRECTIONS = np.array([0.0, 45.0], dtype=np.float32)

# Beam model
BEAM_HALF_WIDTH = 80.0
GAIN_FLOOR = 0.05

# SRP search / frame params
THETA_RESOLUTION = 0.5
FRAME_DURATION = 0.05
FRAME_HOP = 0.025

# Drone band
FREQ_MIN = 50
FREQ_MAX = 6000
