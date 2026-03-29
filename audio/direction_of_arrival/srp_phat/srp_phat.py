"""Backward-compatible entrypoint for the modular SRP-PHAT implementation."""

from __future__ import annotations

from pathlib import Path
import sys


def _ensure_repo_root_on_path() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    root_str = str(repo_root)
    if root_str not in sys.path:
        sys.path.insert(0, root_str)


def main() -> float:
    _ensure_repo_root_on_path()
    from direction_of_arrival.srp_phat.main import main as modular_main

    return modular_main()


if __name__ == "__main__":
    main()
"""
=============================================================================
  Drone Direction of Arrival -- SRP-PHAT with 3 Parabolic Microphones
=============================================================================

Array geometry (top-down view, all distances in meters):

            Mic 1  (points 0 degrees, East)
            *
           / \\
     10cm /   \\ 10cm
         /     \\
        *-------*
      Mic 2    Mic 3
  (points 120)  (points 240)

Each mic is a parabolic dish with a directional beam pattern.
The beam weight reduces gain for sounds arriving off-axis.

Usage:
    python drone_doa.py                          # uses 1.wav, 2.wav, 3.wav
    python drone_doa.py a.wav b.wav c.wav        # custom files
    python drone_doa.py --plot                   # show SRP map plot
    python drone_doa.py --simulate 45            # inject synthetic source at 45 degrees

Algorithm:
    1. Load and synchronise audio from 3 channels
    2. Split into short frames (STFT-style windowing)
    3. Compute GCC-PHAT for every mic pair in each frame
    4. For each candidate direction theta (0->360, 1 degree steps):
         - Compute expected TDOA for each pair given geometry
         - Look up GCC-PHAT value at that TDOA
         - Weight by parabolic beam pattern of both mics
         - Accumulate -> SRP(theta)
    5. Return argmax(SRP) as the estimated DOA
=============================================================================
"""

import sys
import argparse
import struct
import wave
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import resample_poly
from math import gcd


# =============================================================================
# ─── ARRAY CONFIGURATION  (edit these to match your hardware) ────────────────
# =============================================================================

SPEED_OF_SOUND = 343.0  # m/s  (adjust for temperature: ~331 + 0.6*T°C)

# Inter-microphone distance (equilateral triangle, 10 cm sides)
MIC_DISTANCE = 0.50  # metres

# Mic positions in 2D (x, y) — equilateral triangle centred at origin
_R = MIC_DISTANCE / np.sqrt(2)  # circumradius of equilateral triangle
MIC_POSITIONS = np.array(
    [
        [_R * np.cos(np.radians(0)), _R * np.sin(np.radians(0))],  # Mic 1
        [_R * np.cos(np.radians(45)), _R * np.sin(np.radians(45))],  # Mic 2
        # [_R * np.cos(np.radians(330)), _R * np.sin(np.radians(330))],  # Mic 3
    ]
)

# Pointing direction of each parabolic mic (degrees, 0=East, CCW positive)
MIC_DIRECTIONS = np.array([0.0, 45.0])

# Half-beamwidth of each parabolic dish (degrees). Gain drops to 0 at this angle.
# A 30cm dish at 200 Hz gives roughly 20-35° half-beamwidth.
BEAM_HALF_WIDTH = 80.0  # degrees

# Angular resolution of the DOA search grid
THETA_RESOLUTION = 0.5  # degrees  (lower = finer but slower)

# STFT frame settings for SRP accumulation
FRAME_DURATION = 0.05  # seconds per analysis frame
FRAME_HOP = 0.025  # seconds between frames (50% overlap)

# Frequency band to use for DOA (drone rotor harmonics + motor noise)
FREQ_MIN = 50  # Hz
FREQ_MAX = 6000  # Hz


# =============================================================================
# ─── AUDIO I/O ───────────────────────────────────────────────────────────────
# =============================================================================


def load_wav(path: str):
    """Load a WAV file, return (samples_float32, sample_rate)."""
    with wave.open(path, "rb") as wf:
        n_channels = wf.getnchannels()
        sampwidth = wf.getsampwidth()
        fs = wf.getframerate()
        n_frames = wf.getnframes()
        raw = wf.readframes(n_frames)

    # ── Decode raw bytes into int samples ────────────────────────────────────
    if sampwidth == 1:
        # 8-bit unsigned (0..255) → shift to signed
        samples = np.frombuffer(raw, dtype=np.uint8).astype(np.float32) - 128
        scale = 128.0

    elif sampwidth == 2:
        # 16-bit signed — standard CD quality
        samples = np.frombuffer(raw, dtype=np.int16).astype(np.float32)
        scale = 32768.0

    elif sampwidth == 3:
        # 24-bit signed — no native numpy dtype, decode manually.
        # Each sample is 3 little-endian bytes. We pad to 4 bytes and read as int32.
        raw_bytes = np.frombuffer(raw, dtype=np.uint8).reshape(-1, 3)
        # Pad a zero MSB on the left to make 4-byte int32, then shift right by 8
        padded = np.zeros((len(raw_bytes), 4), dtype=np.uint8)
        padded[:, 1:] = raw_bytes  # little-endian: LSB first
        samples = padded.view(np.int32).reshape(-1).astype(np.float32) / 256.0
        scale = 2**23  # 24-bit full scale = 8 388 608

    elif sampwidth == 4:
        # 32-bit signed integer
        samples = np.frombuffer(raw, dtype=np.int32).astype(np.float32)
        scale = 2147483648.0

    else:
        raise ValueError(
            f"Unsupported sample width: {sampwidth} bytes ({sampwidth*8}-bit). "
            f"Supported formats: 8, 16, 24, 32-bit PCM."
        )

    # Normalise to [-1.0, 1.0]
    samples /= scale

    # Deinterleave: keep only the first channel if multichannel
    if n_channels > 1:
        samples = samples[::n_channels]

    return samples, fs


def resample_to(signal: np.ndarray, src_fs: int, dst_fs: int) -> np.ndarray:
    """Resample signal from src_fs to dst_fs using polyphase filter."""
    if src_fs == dst_fs:
        return signal
    g = gcd(src_fs, dst_fs)
    return resample_poly(signal, dst_fs // g, src_fs // g).astype(np.float32)


def load_all(paths):
    """Load all wav files, resample to a common rate, align lengths."""
    loaded = [load_wav(p) for p in paths]
    rates = [r for _, r in loaded]
    target_fs = max(rates)

    signals = []
    for sig, fs in loaded:
        signals.append(resample_to(sig, fs, target_fs))

    # Trim to shortest
    min_len = min(len(s) for s in signals)
    signals = [s[:min_len] for s in signals]

    print(
        f"  Loaded {len(paths)} channels | fs={target_fs} Hz | "
        f"duration={min_len/target_fs:.3f}s | samples={min_len}"
    )
    return np.array(signals), target_fs


# =============================================================================
# ─── BEAM PATTERN ────────────────────────────────────────────────────────────
# =============================================================================


def parabolic_gain(
    theta_source: float, mic_direction: float, half_beamwidth: float = BEAM_HALF_WIDTH
) -> float:
    """
    Simplified parabolic dish gain model.
    Returns a weight in [0, 1] based on the off-axis angle.

    Model: cosine taper from 1 (on-axis) to GAIN_FLOOR (at half_beamwidth and beyond).
    A non-zero floor ensures that even "back" mics still contribute weak TDOA
    evidence, avoiding complete blind spots in the 360-degree coverage.

    Replace with your empirically measured beam pattern for best results.

    Args:
        theta_source:   direction of the source (degrees)
        mic_direction:  pointing direction of the mic (degrees)
        half_beamwidth: half-angle at which gain reaches minimum (degrees)
    """
    GAIN_FLOOR = 0.05  # minimum gain even for off-axis / back-lobe directions
    # models sidelobes and diffraction; set to 0 for ideal hard cutoff
    off_axis = abs((theta_source - mic_direction + 180) % 360 - 180)
    if off_axis >= half_beamwidth:
        return GAIN_FLOOR
    # Cosine taper: 1 at 0 deg, GAIN_FLOOR at half_beamwidth
    main_lobe = np.cos(np.pi * off_axis / (2 * half_beamwidth))
    return float(GAIN_FLOOR + (1.0 - GAIN_FLOOR) * main_lobe)


def pair_weight(theta: float, mic_dir_a: float, mic_dir_b: float) -> float:
    """
    Joint beam weight for a mic pair: geometric mean of both mics gains.
    The floor in parabolic_gain ensures all pairs always contribute at least
    a small amount, preventing hard blind spots in the coverage map.
    """
    g_a = parabolic_gain(theta, mic_dir_a)
    g_b = parabolic_gain(theta, mic_dir_b)
    return float(np.sqrt(g_a * g_b))


# =============================================================================
# ─── GCC-PHAT ────────────────────────────────────────────────────────────────
# =============================================================================


def gcc_phat(
    x1: np.ndarray,
    x2: np.ndarray,
    n_fft: int,
    fs: int,
    freq_min: float = FREQ_MIN,
    freq_max: float = FREQ_MAX,
) -> np.ndarray:
    """
    Compute the GCC-PHAT between two signals in a single frame.

    Returns the cross-correlation in the time domain (length n_fft).
    The peak of this array occurs at the lag (in samples) = TDOA * fs.

    Steps:
      1. FFT both signals
      2. Compute cross-spectrum
      3. Apply PHAT whitening  (keep only phase, discard magnitude)
      4. Apply bandpass mask   (only use frequencies relevant to drones)
      5. IFFT → time-domain correlation
    """
    X1 = np.fft.rfft(x1, n=n_fft)
    X2 = np.fft.rfft(x2, n=n_fft)

    # Cross-spectrum
    cross = X1 * np.conj(X2)

    # PHAT whitening: normalise magnitude, keep phase
    cross /= np.abs(cross) + 1e-8

    # Bandpass: zero out frequencies outside [freq_min, freq_max]
    freqs = np.fft.rfftfreq(n_fft, d=1.0 / fs)
    band_mask = (freqs >= freq_min) & (freqs <= freq_max)
    cross[~band_mask] = 0.0

    # Back to time domain
    gcc = np.fft.irfft(cross, n=n_fft)
    return gcc.real


def tdoa_to_samples(tdoa_seconds: float, fs: int, n_fft: int) -> int:
    """Convert a TDOA in seconds to a circular index into the GCC array."""
    return int(round(tdoa_seconds * fs)) % n_fft


# =============================================================================
# ─── SRP-PHAT CORE ───────────────────────────────────────────────────────────
# =============================================================================


def compute_srp_map(
    signals: np.ndarray,
    mic_positions: np.ndarray,
    mic_directions: np.ndarray,
    fs: int,
    theta_grid: np.ndarray,
) -> np.ndarray:
    """
    Compute the SRP-PHAT map over all frames and accumulate.

    For each frame:
      - Compute GCC-PHAT for all mic pairs
      - For each θ in theta_grid:
          srp(θ) += Σ_{pairs} weight(θ) * gcc[expected_tdoa(θ)]
    """
    n_mics = signals.shape[0]
    n_samples = signals.shape[1]
    n_fft = int(FRAME_DURATION * fs)
    hop = int(FRAME_HOP * fs)
    pairs = [(i, j) for i in range(n_mics) for j in range(i + 1, n_mics)]

    srp_accumulated = np.zeros(len(theta_grid))
    n_frames = 0

    # ── Precompute expected TDOA (samples) for every pair × every θ ──────────
    print(
        f"  Precomputing TDOA table  "
        f"({len(pairs)} pairs × {len(theta_grid)} directions)..."
    )

    tdoa_table = np.zeros((len(pairs), len(theta_grid)), dtype=int)
    weight_table = np.zeros((len(pairs), len(theta_grid)))

    for p_idx, (i, j) in enumerate(pairs):
        for t_idx, theta in enumerate(theta_grid):
            # Unit vector pointing in direction θ
            u = np.array([np.cos(np.radians(theta)), np.sin(np.radians(theta))])

            # Expected TDOA: projection of baseline onto source direction
            baseline = mic_positions[i] - mic_positions[j]
            tau = np.dot(baseline, u) / SPEED_OF_SOUND  # seconds

            tdoa_table[p_idx, t_idx] = tdoa_to_samples(-tau, fs, n_fft)
            weight_table[p_idx, t_idx] = pair_weight(
                theta, mic_directions[i], mic_directions[j]
            )

    # ── Frame-by-frame processing ─────────────────────────────────────────────
    print(f"  Processing frames  (frame={n_fft} samples, hop={hop} samples)...")

    frame_start = 0
    while frame_start + n_fft <= n_samples:
        frame_end = frame_start + n_fft

        # Window each mic's frame
        window = np.hanning(n_fft)
        frames = [signals[m, frame_start:frame_end] * window for m in range(n_mics)]

        # GCC-PHAT for each pair
        gccs = {}
        for p_idx, (i, j) in enumerate(pairs):
            gccs[p_idx] = gcc_phat(frames[i], frames[j], n_fft, fs)

        # Accumulate SRP map
        srp_frame = np.zeros(len(theta_grid))
        for p_idx in range(len(pairs)):
            gcc_vals = gccs[p_idx][tdoa_table[p_idx]]  # vectorised lookup
            srp_frame += weight_table[p_idx] * gcc_vals

        srp_accumulated += srp_frame
        n_frames += 1
        frame_start += hop

    print(f"  Processed {n_frames} frames")
    return srp_accumulated / max(n_frames, 1)


# =============================================================================
# ─── PEAK REFINEMENT ─────────────────────────────────────────────────────────
# =============================================================================


def refine_peak(srp_map: np.ndarray, theta_grid: np.ndarray) -> float:
    """
    Sub-grid peak refinement via parabolic interpolation.
    Finds the true peak between grid points for better angular precision.
    """
    idx = int(np.argmax(srp_map))
    n = len(srp_map)

    # Wrap-around neighbours
    idx_prev = (idx - 1) % n
    idx_next = (idx + 1) % n

    y0 = srp_map[idx_prev]
    y1 = srp_map[idx]
    y2 = srp_map[idx_next]

    denom = 2 * (2 * y1 - y0 - y2)
    if abs(denom) < 1e-12:
        return float(theta_grid[idx])

    delta = (y0 - y2) / denom  # fractional shift in grid steps
    step = theta_grid[1] - theta_grid[0]
    return float((theta_grid[idx] + delta * step) % 360)


def find_top_k_peaks(
    srp_map: np.ndarray,
    theta_grid: np.ndarray,
    k: int = 3,
    min_separation_deg: float = 30.0,
):
    """
    Find the top-k peaks in the SRP map.
    Useful when multiple drones are present.
    """
    map_copy = srp_map.copy()
    peaks = []
    step = theta_grid[1] - theta_grid[0]
    sep_bins = int(min_separation_deg / step)

    for _ in range(k):
        idx = int(np.argmax(map_copy))
        peaks.append((float(theta_grid[idx]), float(map_copy[idx])))

        # Suppress neighbourhood around found peak
        for offset in range(-sep_bins, sep_bins + 1):
            map_copy[(idx + offset) % len(map_copy)] = 0.0

    return peaks


# =============================================================================
# ─── SIMULATION (synthetic test without real WAV files) ──────────────────────
# =============================================================================


def simulate_signals(
    source_angle_deg: float,
    fs: int = 44100,
    duration: float = 1.0,
    snr_db: float = 20.0,
    mic_positions: np.ndarray = MIC_POSITIONS,
) -> np.ndarray:
    """
    Generate synthetic microphone signals for a drone at a known angle.

    The drone signal is a mix of rotor harmonics (80, 160, 240 Hz)
    plus broadband noise, propagated to each mic with the correct TDOA.
    """
    n_samples = int(duration * fs)
    t = np.arange(n_samples) / fs

    # Synthetic drone: fundamental + harmonics (typical quadrotor ~80 Hz)
    drone = (
        np.sin(2 * np.pi * 80 * t) * 0.5
        + np.sin(2 * np.pi * 160 * t) * 0.3
        + np.sin(2 * np.pi * 240 * t) * 0.2
        + np.random.randn(n_samples) * 0.1
    )  # broadband component

    u = np.array(
        [np.cos(np.radians(source_angle_deg)), np.sin(np.radians(source_angle_deg))]
    )

    signals = []
    noise_power = 10 ** (-snr_db / 10)

    for pos in mic_positions:
        # TDOA relative to array centre
        tau = -np.dot(pos, u) / SPEED_OF_SOUND
        delay_s = int(round(tau * fs))

        # Apply delay (circular shift)
        delayed = np.roll(drone, delay_s)

        # Add mic noise
        noise = np.random.randn(n_samples) * np.sqrt(noise_power)
        signals.append(delayed + noise)

    return np.array(signals, dtype=np.float32)


def save_simulation_wavs(
    signals: np.ndarray,
    fs: int,
    source_angle_deg: float,
    out_dir: str = "simulation_signals",
) -> list:
    """
    Save each simulated microphone signal as a 16-bit PCM WAV file.

    Files are written to `out_dir/` and named:
        mic1_<angle>deg.wav
        mic2_<angle>deg.wav
        ...

    Args:
        signals:          array of shape (n_mics, n_samples)
        fs:               sample rate in Hz
        source_angle_deg: the simulated source angle (used in filename)
        out_dir:          output directory (created if it does not exist)

    Returns:
        List of file paths written.
    """
    import os

    os.makedirs(out_dir, exist_ok=True)

    paths = []
    for i, sig in enumerate(signals):
        filename = f"mic{i+1}_{int(source_angle_deg):03d}deg.wav"
        filepath = os.path.join(out_dir, filename)

        # Clip and convert to 16-bit PCM
        sig_clipped = np.clip(sig, -1.0, 1.0)
        pcm = (sig_clipped * 32767).astype(np.int16)

        with wave.open(filepath, "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)  # 16-bit = 2 bytes
            wf.setframerate(fs)
            wf.writeframes(pcm.tobytes())

        paths.append(filepath)

    return paths


# =============================================================================
# ─── REPORTING & PLOTTING ────────────────────────────────────────────────────
# =============================================================================


def compass_label(deg: float) -> str:
    """Convert degrees to compass label (N/NE/E/...)."""
    dirs = [
        "E",
        "ENE",
        "NE",
        "NNE",
        "N",
        "NNW",
        "NW",
        "WNW",
        "W",
        "WSW",
        "SW",
        "SSW",
        "S",
        "SSE",
        "SE",
        "ESE",
    ]
    idx = int((deg + 11.25) / 22.5) % 16
    return dirs[idx]


def print_report(estimated_doa: float, peaks, srp_map, theta_grid):
    print("\n" + "═" * 60)
    print("  DRONE DIRECTION OF ARRIVAL — RESULTS")
    print("═" * 60)
    print(
        f"  Primary DOA  :  {estimated_doa:6.1f}°  " f"({compass_label(estimated_doa)})"
    )
    print()
    print("  Top candidate directions:")
    for rank, (angle, power) in enumerate(peaks, 1):
        max_power = max(p for _, p in peaks) or 1.0
        bar = "█" * int(power / max_power * 20)
        print(
            f"    #{rank}  {angle:6.1f}°  ({compass_label(angle):3s})  "
            f"power={power:.4f}  {bar}"
        )
    print("═" * 60 + "\n")


def plot_srp_map(
    srp_map: np.ndarray,
    theta_grid: np.ndarray,
    estimated_doa: float,
    mic_directions: np.ndarray = MIC_DIRECTIONS,
    true_angle: float = None,
    save_path: str = "srp_map.png",
):
    """Plot the SRP map as a polar diagram and a linear plot."""
    theta_rad = np.radians(theta_grid)
    norm_srp = srp_map / (np.max(srp_map) + 1e-12)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6), subplot_kw={"projection": None})
    fig.suptitle(
        "SRP-PHAT — Drone Direction of Arrival", fontsize=14, fontweight="bold"
    )

    # ── Polar plot ────────────────────────────────────────────────────────────
    ax_polar = fig.add_subplot(121, projection="polar")
    ax_polar.plot(
        theta_rad, norm_srp, color="royalblue", linewidth=1.5, label="SRP-PHAT"
    )
    ax_polar.fill(theta_rad, norm_srp, alpha=0.15, color="royalblue")

    # Mark estimated DOA
    est_rad = np.radians(estimated_doa)
    ax_polar.annotate(
        "",
        xy=(est_rad, 1.05),
        xytext=(est_rad, 0.0),
        arrowprops=dict(arrowstyle="->", color="red", lw=2),
    )
    ax_polar.plot(
        est_rad,
        norm_srp[int(estimated_doa / THETA_RESOLUTION)],
        "ro",
        markersize=8,
        label=f"DOA = {estimated_doa:.1f}°",
    )

    # Mark true angle if simulating
    if true_angle is not None:
        true_rad = np.radians(true_angle)
        ax_polar.axvline(
            true_rad,
            color="green",
            linestyle="--",
            linewidth=1.5,
            label=f"True = {true_angle:.1f}°",
        )

    # Mark mic pointing directions
    for k, mic_dir in enumerate(mic_directions):
        mrad = np.radians(mic_dir)
        ax_polar.plot([mrad, mrad], [0, 0.3], color="orange", linewidth=2, alpha=0.7)
        ax_polar.text(
            mrad,
            0.35,
            f"M{k+1}",
            ha="center",
            va="center",
            fontsize=8,
            color="darkorange",
        )

    ax_polar.set_theta_zero_location("E")
    ax_polar.set_theta_direction(1)
    ax_polar.set_rticks([0.25, 0.5, 0.75, 1.0])
    ax_polar.set_rlabel_position(45)
    ax_polar.legend(loc="lower right", fontsize=8)
    ax_polar.set_title("Polar SRP Map", pad=20)

    # ── Linear plot ───────────────────────────────────────────────────────────
    ax_lin = axes[1]
    ax_lin.plot(theta_grid, norm_srp, color="royalblue", linewidth=1.5)
    ax_lin.fill_between(theta_grid, 0, norm_srp, alpha=0.15, color="royalblue")
    ax_lin.axvline(
        estimated_doa,
        color="red",
        linewidth=2,
        linestyle="-",
        label=f"Estimated DOA = {estimated_doa:.1f}°",
    )
    if true_angle is not None:
        ax_lin.axvline(
            true_angle,
            color="green",
            linewidth=2,
            linestyle="--",
            label=f"True = {true_angle:.1f}°",
        )

    for k, mic_dir in enumerate(mic_directions):
        ax_lin.axvline(
            mic_dir,
            color="orange",
            linewidth=1,
            linestyle=":",
            alpha=0.7,
            label=f"Mic {k+1} ({mic_dir:.0f}°)",
        )

    ax_lin.set_xlabel("Direction θ (degrees)")
    ax_lin.set_ylabel("Normalised SRP")
    ax_lin.set_xlim(0, 360)
    ax_lin.set_xticks(range(0, 361, 30))
    ax_lin.set_ylim(0, 1.1)
    ax_lin.grid(True, alpha=0.3)
    ax_lin.legend(fontsize=8)
    ax_lin.set_title("Linear SRP Map")

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"  Plot saved → {save_path}")
    plt.show()


# =============================================================================
# ─── MAIN ────────────────────────────────────────────────────────────────────
# =============================================================================


def main():
    parser = argparse.ArgumentParser(
        description="SRP-PHAT Drone DOA with 3 parabolic microphones"
    )
    parser.add_argument(
        "wav_files",
        nargs="*",
        default=["1.wav", "2.wav", "3.wav"],
        help="WAV file per mic (default: 1.wav 2.wav 3.wav)",
    )
    parser.add_argument(
        "--plot", action="store_true", help="Show and save SRP polar map"
    )
    parser.add_argument(
        "--simulate",
        type=float,
        default=None,
        metavar="DEG",
        help="Skip WAV files, simulate a drone at DEG degrees",
    )
    parser.add_argument(
        "--snr", type=float, default=20.0, help="SNR (dB) for simulation (default: 20)"
    )
    parser.add_argument(
        "--top", type=int, default=3, help="Number of candidate directions to report"
    )
    parser.add_argument(
        "--save-sim",
        action="store_true",
        help="Save simulated mic signals as WAV files in ./simulation_signals/",
    )
    args = parser.parse_args()

    print("\n╔══════════════════════════════════════════════╗")
    print("║   Drone DOA — SRP-PHAT  (3 parabolic mics)  ║")
    print("╚══════════════════════════════════════════════╝\n")

    # ── Load or simulate signals ──────────────────────────────────────────────
    true_angle = None
    if args.simulate is not None:
        true_angle = args.simulate % 360
        print(f"[SIMULATION] Source at {true_angle:.1f}°, SNR={args.snr} dB\n")
        fs = 44100
        signals = simulate_signals(
            true_angle, fs=fs, snr_db=args.snr, mic_positions=MIC_POSITIONS
        )

        if args.save_sim:
            print("[SAVING SIMULATION WAV FILES]")
            saved = save_simulation_wavs(signals, fs, true_angle)
            for p in saved:
                print(f"  Wrote: {p}")
            print()
    else:
        if len(args.wav_files) != 2:
            print("ERROR: Provide exactly 2 WAV files (one per mic).")
            sys.exit(1)
        print("[LOADING WAV FILES]")
        for i, f in enumerate(args.wav_files):
            print(f"  Mic {i+1}: {f}")
        print()
        signals, fs = load_all(args.wav_files)

    # ── Angular search grid ───────────────────────────────────────────────────
    theta_grid = np.arange(0.0, 360.0, THETA_RESOLUTION)

    # ── Compute SRP map ───────────────────────────────────────────────────────
    print("\n[SRP-PHAT COMPUTATION]")
    srp_map = compute_srp_map(signals, MIC_POSITIONS, MIC_DIRECTIONS, fs, theta_grid)

    # ── Find DOA ──────────────────────────────────────────────────────────────
    estimated_doa = refine_peak(srp_map, theta_grid)
    peaks = find_top_k_peaks(srp_map, theta_grid, k=args.top)

    print_report(estimated_doa, peaks, srp_map, theta_grid)

    if true_angle is not None:
        error = abs((estimated_doa - true_angle + 180) % 360 - 180)
        print(f"  [SIMULATION ACCURACY]  Error = {error:.2f}°\n")

    # ── Plot ──────────────────────────────────────────────────────────────────
    if args.plot or args.simulate is not None:
        plot_srp_map(
            srp_map,
            theta_grid,
            estimated_doa,
            mic_directions=MIC_DIRECTIONS,
            true_angle=true_angle,
            save_path="srp_map.png",
        )

    return estimated_doa


if __name__ == "__main__":
    main()
