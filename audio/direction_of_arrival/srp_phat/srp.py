import argparse
import wave
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import fftconvolve

fs = 48000
SPEED_OF_SOUND = 343.0
MIC_DISTANCE = 0.50  # meters
MIC1_ANGLE = 0  # degrees, pointing direction of mic 1
MIC2_ANGLE = 45  # degrees, pointing direction of mic 2


def load_wav(file_path):
    with wave.open(file_path) as f:
        samples = f.getnframes()
        audio = f.readframes(samples)
    audio_np = np.frombuffer(audio, dtype=np.int16).astype(np.float32)
    return audio_np / (2**15)


def gcc_phat(sig1, sig2, fs, max_tau=None):
    """
    Generalized Cross-Correlation with Phase Transform (GCC-PHAT).
    Returns estimated TDOA in seconds (sig2 leads if positive).
    """
    n = len(sig1) + len(sig2) - 1
    # Next power of 2 for speed
    n_fft = 2 ** int(np.ceil(np.log2(n)))

    S1 = np.fft.rfft(sig1, n=n_fft)
    S2 = np.fft.rfft(sig2, n=n_fft)

    # Cross-spectrum, PHAT-weighted (whitened phase)
    cross = S1 * np.conj(S2)
    cross /= np.abs(cross) + 1e-10

    cc = np.fft.irfft(cross, n=n_fft)

    # Shift so zero-lag is in center
    cc = np.fft.fftshift(cc)
    lags = np.arange(-n_fft // 2, n_fft // 2) / fs

    # Restrict search range to physically plausible delays
    if max_tau is None:
        max_tau = MIC_DISTANCE / SPEED_OF_SOUND

    mask = np.abs(lags) <= max_tau
    cc_masked = cc * mask

    peak_idx = np.argmax(cc_masked)
    tau = lags[peak_idx]
    confidence = cc_masked[peak_idx] / (np.max(np.abs(cc_masked)) + 1e-10)

    return tau, lags, cc_masked, confidence


def tdoa_to_doa(tau):
    """
    Convert TDOA (seconds) to Direction of Arrival (degrees).

    τ = (d / c) * cos(θ)  →  θ = arccos(τ * c / d)

    θ is the angle from the baseline axis between the two mics.
    Returns two candidate angles (the ambiguity inherent to a 2-mic array).
    """
    ratio = tau * SPEED_OF_SOUND / MIC_DISTANCE
    ratio = np.clip(ratio, -1.0, 1.0)  # numerical safety
    theta = np.degrees(np.arccos(ratio))
    return theta, 360 - theta  # front/back ambiguity


def amplitude_based_doa(sig1, sig2):
    """
    Use RMS amplitude ratio to help resolve ambiguity.
    Each parabolic mic has higher gain in its pointing direction.
    A louder signal on mic1 → source closer to 0°,
    louder on mic2 → source closer to 30°.
    """
    rms1 = np.sqrt(np.mean(sig1**2))
    rms2 = np.sqrt(np.mean(sig2**2))
    ratio_db = 20 * np.log10(rms1 / (rms2 + 1e-10))
    print(f"  Mic1 RMS: {rms1:.4f}, Mic2 RMS: {rms2:.4f}")
    print(f"  Amplitude ratio (Mic1/Mic2): {ratio_db:+.1f} dB")
    if ratio_db > 1:
        print("  → Source likely closer to Mic1 direction (0°)")
    elif ratio_db < -1:
        print("  → Source likely closer to Mic2 direction (30°)")
    else:
        print("  → Source roughly equidistant from both mic directions")
    return ratio_db


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--plot", action="store_true")
    parser.add_argument("--files", nargs="*")
    args = parser.parse_args()

    if not args.files or len(args.files) < 2:
        print("Please provide exactly 2 audio files: --files mic1.wav mic2.wav")
        return

    print("Loading files...")
    sig1 = load_wav(args.files[0])
    sig2 = load_wav(args.files[1])

    # Trim to same length
    min_len = min(len(sig1), len(sig2))
    sig1, sig2 = sig1[:min_len], sig2[:min_len]

    print("\n--- GCC-PHAT TDOA Estimation ---")
    tau, lags, cc, confidence = gcc_phat(sig1, sig2, fs)
    print(f"  Estimated TDOA: {tau * 1000:.3f} ms")
    print(f"  Confidence (peak sharpness): {confidence:.3f}")

    print("\n--- Direction of Arrival ---")
    angle1, angle2 = tdoa_to_doa(tau)
    print(f"  Candidate DOA angles: {angle1:.1f}° or {angle2:.1f}° (from baseline)")

    print("\n--- Amplitude Analysis (ambiguity resolution) ---")
    amplitude_based_doa(sig1, sig2)

    if args.plot:
        fig, axes = plt.subplots(2, 1, figsize=(12, 6))

        t = np.arange(min_len) / fs
        axes[0].plot(t, sig1, alpha=0.7, label=f"Mic1 (0°)")
        axes[0].plot(t, sig2, alpha=0.7, label=f"Mic2 (30°)")
        axes[0].set_xlabel("Time (s)")
        axes[0].set_ylabel("Amplitude")
        axes[0].set_title("Waveforms")
        axes[0].legend()

        axes[1].plot(lags * 1000, cc)
        axes[1].axvline(
            tau * 1000, color="r", linestyle="--", label=f"Peak τ = {tau*1000:.3f} ms"
        )
        axes[1].set_xlabel("Lag (ms)")
        axes[1].set_ylabel("GCC-PHAT")
        axes[1].set_title("Cross-Correlation (GCC-PHAT)")
        axes[1].legend()

        plt.tight_layout()
        plt.savefig("doa_result.png", dpi=150)
        plt.show()


if __name__ == "__main__":
    main()
