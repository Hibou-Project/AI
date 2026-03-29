import argparse
import sys
import wave

import numpy as np


def load_wav(file_path: str) -> tuple[np.ndarray, int]:
    try:
        with wave.open(file_path, "rb") as f:
            n_frames = f.getnframes()
            sampwidth = f.getsampwidth()
            rate = f.getframerate()
            channels = f.getnchannels()
            raw = f.readframes(n_frames)
    except (FileNotFoundError, wave.Error) as e:
        raise RuntimeError(f"Failed to load '{file_path}': {e}") from e

    if sampwidth == 2:
        audio = np.frombuffer(raw, dtype=np.int16).astype(np.float32)
        audio /= 32768.0
    elif sampwidth == 3:
        raw_bytes = np.frombuffer(raw, dtype=np.uint8).reshape(-1, 3)
        padded = np.zeros((raw_bytes.shape[0], 4), dtype=np.uint8)
        padded[:, 1:] = raw_bytes
        audio = padded.view(np.int32).flatten().astype(np.float32)
        audio /= 2147483648.0
    else:
        raise ValueError(
            f"Unsupported sample width: {sampwidth} bytes in '{file_path}'"
        )

    if channels > 1:
        raise NotImplementedError("Multi-channel audio not supported")

    return audio, rate


def equalize_audio_size(audios: list[np.ndarray]) -> list[np.ndarray]:
    min_len = min(a.shape[0] for a in audios)
    return [a[:min_len] for a in audios]


def rms_frames(signal: np.ndarray, frame_size: int, hop_size: int) -> np.ndarray:
    """Compute per-frame RMS energy using stride tricks (zero-copy)."""
    n_frames = 1 + (len(signal) - frame_size) // hop_size
    strides = (signal.strides[0] * hop_size, signal.strides[0])
    frames = np.lib.stride_tricks.as_strided(
        signal, shape=(n_frames, frame_size), strides=strides
    )
    return np.sqrt(np.mean(frames**2, axis=1))


def amplitude_doa(
    sig_left: np.ndarray,
    sig_right: np.ndarray,
    fs: int,
    frame_ms: float = 50.0,
    hop_ms: float = 25.0,
    energy_gate_ratio: float = 0.10,
) -> dict:
    """
    Estimate Direction of Arrival from the amplitude (power) difference
    between two parabolic microphones pointing in opposite directions.

    Assumes a cardioid-like gain model  G(α) = (1 + cos α) / 2  for each
    dish, which yields  NPD ≈ sin(θ)  ⇒  θ = arcsin(NPD).

    Convention
    ----------
        sig_left  = mic pointing LEFT  (look direction −90°)
        sig_right = mic pointing RIGHT (look direction +90°)
        θ = 0°    → source straight ahead  (broadside)
        θ < 0     → source to the LEFT
        θ > 0     → source to the RIGHT

    Parameters
    ----------
    energy_gate_ratio : float
        Frames whose total energy is below this fraction of the peak are
        excluded from the global (weighted) estimate.
    """
    frame_size = int(fs * frame_ms / 1000)
    hop_size = int(fs * hop_ms / 1000)

    rms_l = rms_frames(sig_left, frame_size, hop_size)
    rms_r = rms_frames(sig_right, frame_size, hop_size)

    eps = 1e-10
    total = rms_l + rms_r + eps

    # Normalised Power Difference ∈ [−1, 1]
    npd = (rms_r - rms_l) / total

    # ILD in dB (informational)
    ild_db = 20.0 * np.log10((rms_r + eps) / (rms_l + eps))

    # Map NPD → angle via arcsin (matches cardioid gain model)
    theta_per_frame = np.degrees(np.arcsin(np.clip(npd, -1.0, 1.0)))

    # Energy-weighted global estimate, gating out silence
    energy_threshold = energy_gate_ratio * np.max(total)
    active = total > energy_threshold

    if np.any(active):
        weights = total[active]
        theta_global = float(np.average(theta_per_frame[active], weights=weights))
    else:
        theta_global = 0.0

    times = np.arange(len(rms_l)) * hop_size / fs

    return {
        "theta_global_deg": theta_global,
        "theta_per_frame_deg": theta_per_frame,
        "ild_db": ild_db,
        "npd": npd,
        "rms_left": rms_l,
        "rms_right": rms_r,
        "times": times,
        "active_mask": active,
    }


def plot_results(result: dict, save_path: str | None = None) -> None:
    import matplotlib.pyplot as plt

    t = result["times"]

    fig, axes = plt.subplots(3, 1, figsize=(12, 8), sharex=True)

    # --- RMS levels ---
    ax = axes[0]
    ax.plot(t, 20 * np.log10(result["rms_left"] + 1e-10), label="Mic LEFT", alpha=0.8)
    ax.plot(
        t, 20 * np.log10(result["rms_right"] + 1e-10), label="Mic RIGHT", alpha=0.8
    )
    ax.set_ylabel("RMS (dBFS)")
    ax.legend()
    ax.set_title("Amplitude-Based DOA — Two Parabolic Microphones")
    ax.grid(True, alpha=0.3)

    # --- ILD ---
    ax = axes[1]
    ax.plot(t, result["ild_db"], color="purple", alpha=0.8)
    ax.axhline(0, color="gray", ls="--", lw=0.5)
    ax.set_ylabel("ILD (dB)\n← left | right →")
    ax.grid(True, alpha=0.3)

    # --- DOA angle ---
    ax = axes[2]
    theta = np.where(result["active_mask"], result["theta_per_frame_deg"], np.nan)
    ax.plot(t, theta, color="tab:red", alpha=0.8, label="Per-frame DOA")
    ax.axhline(
        result["theta_global_deg"],
        color="black",
        ls="--",
        lw=1.5,
        label=f'Global: {result["theta_global_deg"]:+.1f}°',
    )
    ax.set_ylabel("θ (deg)\n← left | right →")
    ax.set_xlabel("Time (s)")
    ax.set_ylim(-95, 95)
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
        print(f"  Plot saved to: {save_path}")
    plt.show()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Amplitude-based DOA for two parabolic microphones"
    )
    parser.add_argument(
        "wav_files",
        nargs=2,
        metavar="WAV",
        help="mic_left.wav mic_right.wav (left-pointing then right-pointing)",
    )
    parser.add_argument(
        "--frame-ms",
        type=float,
        default=50.0,
        help="Analysis frame length in ms (default: 50)",
    )
    parser.add_argument(
        "--hop-ms",
        type=float,
        default=25.0,
        help="Hop between frames in ms (default: 25)",
    )
    parser.add_argument(
        "--gate",
        type=float,
        default=0.10,
        help="Energy gate: ignore frames below this fraction of peak (default: 0.10)",
    )
    parser.add_argument(
        "--plot",
        nargs="?",
        const="doa_amplitude.png",
        default=None,
        metavar="FILE",
        help="Save a diagnostic plot (default filename: doa_amplitude.png)",
    )

    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)

    args = parser.parse_args()

    print("Loading WAV files...")
    signals = []
    for path in args.wav_files:
        audio, rate = load_wav(path)
        signals.append(audio)
        print(f"  {path}  |  {rate} Hz  |  {len(audio)} samples")

    signals = equalize_audio_size(signals)
    sig_left, sig_right = signals

    result = amplitude_doa(
        sig_left,
        sig_right,
        rate,
        frame_ms=args.frame_ms,
        hop_ms=args.hop_ms,
        energy_gate_ratio=args.gate,
    )

    theta = result["theta_global_deg"]
    if theta < -5:
        direction = "LEFT"
    elif theta > 5:
        direction = "RIGHT"
    else:
        direction = "CENTER"

    print(f"\n  Estimated DOA : {theta:+.1f}°  ({direction})")
    print(f"  Mean ILD      : {np.mean(result['ild_db'][result['active_mask']]):+.1f} dB")
    print(f"  Active frames : {np.sum(result['active_mask'])} / {len(result['active_mask'])}")
    print(f"  Convention    : 0° = broadside, −90° = left, +90° = right")

    if args.plot is not None:
        plot_results(result, save_path=args.plot)


if __name__ == "__main__":
    main()
