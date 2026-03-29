import argparse
import json
import sys
import wave
import numpy as np
import pyaudio
from pyroomacoustics.experimental.localization import tdoa, tdoa_loc


def play_audio(audio: np.ndarray, sr: int) -> None:
    p = pyaudio.PyAudio()
    stream = p.open(format=pyaudio.paFloat32, channels=1, rate=sr, output=True)
    try:
        stream.write(audio.tobytes())
    finally:
        stream.stop_stream()
        stream.close()
        p.terminate()


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
    elif sampwidth == 4:
        try:
            audio = np.frombuffer(raw, dtype=np.float32)
        except ValueError:
            audio = np.frombuffer(raw, dtype=np.int32).astype(np.float32)
            audio /= 2147483648.0
    else:
        raise ValueError(f"Unsupported sample width: {sampwidth} bytes")

    if channels > 1:
        raise NotImplementedError("Multi-channel WAV detected. Please provide mono files.")

    return audio, rate


def equalize_audio_size(audios: list[np.ndarray]) -> list[np.ndarray]:
    min_len = min(audio.shape[0] for audio in audios)
    return [audio[:min_len] for audio in audios]


def compute_tdoa_vector(
    signals: list[np.ndarray], start: int, length: int, interp: int, fs: int
) -> np.ndarray:
    """Compute TDOA of each mic relative to mic 0 for a given chunk."""
    n_mics = len(signals)
    tdoas = np.zeros(n_mics)
    ref_chunk = signals[0][start : start + length]
    for j in range(1, n_mics):
        tdoas[j] = tdoa(ref_chunk, signals[j][start : start + length], interp=interp, fs=fs, phat=True)
    return tdoas


# ── 2-mic mode ───────────────────────────────────────────────────────────────

def run_2mic(signals, rate, args):
    """TDOA angle estimation for a 2-microphone pair (e.g. parabolic mics)."""
    c = args.speed_of_sound
    L = args.distance
    bearing_offset = args.array_bearing
    chunk_samples = int(rate * args.chunk)
    total_samples = len(signals[0])
    step = chunk_samples

    print(f"Mode: 2-mic TDOA  |  distance: {L} m  |  bearing offset: {bearing_offset}°")
    print(f"Fs: {rate} Hz  |  c: {c} m/s")
    print(f"Processing {total_samples / rate:.2f}s audio in {args.chunk}s chunks...")
    sep = "-" * 60
    print(sep)
    print(f"{'Time (s)':<10} | {'Angle (deg)':<12} | {'Delay (ms)':<12} | Status")
    print(sep)

    for i in range(0, total_samples - chunk_samples, step):
        s1 = signals[0][i : i + chunk_samples]
        s2 = signals[1][i : i + chunk_samples]

        if np.sqrt(np.mean(s1**2)) < 0.01:
            print(f"{i / rate:<10.2f} | {'---':<12} | {'---':<12} | Silent")
            continue

        tau = tdoa(s1, s2, interp=args.interp, fs=rate, phat=True)

        ratio = np.clip(c * tau / L, -1.0, 1.0)
        theta = np.degrees(np.arcsin(ratio)) + bearing_offset

        print(f"{i / rate:<10.2f} | {theta:<12.2f} | {tau * 1000:<12.4f} | OK")

    print(sep)
    print("Done.")


# ── 3+ mic mode ──────────────────────────────────────────────────────────────

def run_multimicmic(signals, rate, mic_positions, args):
    """Full 3-D localization via tdoa_loc for >= 3 microphones."""
    c = args.speed_of_sound
    n_mics = len(signals)
    R = mic_positions.T  # tdoa_loc expects 3×N
    chunk_samples = int(rate * args.chunk)
    total_samples = len(signals[0])
    step = chunk_samples

    print(f"Mode: tdoa_loc ({n_mics} mics)  |  Fs: {rate} Hz  |  c: {c} m/s")
    print(f"Processing {total_samples / rate:.2f}s audio in {args.chunk}s chunks...")
    sep = "-" * 72
    print(sep)
    print(f"{'Time (s)':<10} | {'X (m)':<9} | {'Y (m)':<9} | {'Z (m)':<9} | {'Azimuth (deg)':<14} | Status")
    print(sep)

    for i in range(0, total_samples - chunk_samples, step):
        ref_chunk = signals[0][i : i + chunk_samples]
        if np.sqrt(np.mean(ref_chunk**2)) < 0.01:
            print(f"{i / rate:<10.2f} | {'---':<9} | {'---':<9} | {'---':<9} | {'---':<14} | Silent")
            continue

        tdoas = compute_tdoa_vector(signals, i, chunk_samples, args.interp, rate)

        try:
            loc = tdoa_loc(R, tdoas, c)
            x, y, z = loc.flatten()[:3]
            azimuth = np.degrees(np.arctan2(y, x))
            print(f"{i / rate:<10.2f} | {x:<9.3f} | {y:<9.3f} | {z:<9.3f} | {azimuth:<14.2f} | OK")
        except Exception as e:
            if args.debug:
                print(f"{i / rate:<10.2f} | {'err':<9} | {'err':<9} | {'err':<9} | {'err':<14} | {e}")
            else:
                print(f"{i / rate:<10.2f} | {'err':<9} | {'err':<9} | {'err':<9} | {'err':<14} | Failed")

    print(sep)
    print("Done.")


# ── CLI ───────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="pyroomacoustics TDOA-based DOA / localization on audio chunks",
        epilog=(
            "2-mic example (parabolic mics, 30° apart):\n"
            "  python pyroomtdoa.py left.wav right.wav --distance 0.5 --array-bearing 15\n\n"
            "Multi-mic example (>= 3 mics, full 3-D localization):\n"
            '  python pyroomtdoa.py m1.wav m2.wav m3.wav m4.wav \\\n'
            '    --mic-positions "[[0,0,0],[0.18,0,0],[0,0.18,0],[0.18,0.18,0]]"'
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("wav_files", nargs="+", help="One mono WAV file per microphone (>= 2)")

    g2 = parser.add_argument_group("2-mic options")
    g2.add_argument("--distance", type=float, default=0.5, help="Distance between the two mics in metres (default: 0.5)")
    g2.add_argument(
        "--array-bearing", type=float, default=0.0,
        help="Bearing offset of the array broadside in degrees (default: 0). "
             "Added to the raw TDOA angle so that 0° = the midpoint between the two parabolic dishes.",
    )

    gm = parser.add_argument_group("multi-mic options (>= 3)")
    gm.add_argument(
        "--mic-positions",
        help='JSON array of [x,y,z] positions in metres, e.g. "[[0,0,0],[0.5,0,0],[0,0.5,0]]"',
    )

    gc = parser.add_argument_group("common options")
    gc.add_argument("--chunk", type=float, default=0.5, help="Chunk size in seconds (default: 0.5)")
    gc.add_argument("--interp", type=int, default=16, help="GCC-PHAT interpolation factor (default: 16)")
    gc.add_argument("--speed-of-sound", type=float, default=343.0, help="Speed of sound m/s (default: 343)")
    gc.add_argument("--debug", action="store_true")

    args = parser.parse_args()
    n_mics = len(args.wav_files)

    if n_mics < 2:
        print("Error: provide at least 2 WAV files.", file=sys.stderr)
        sys.exit(1)

    # --- Load audio ---
    print("Loading files...")
    signals: list[np.ndarray] = []
    rate = None
    for path in args.wav_files:
        try:
            audio, sr = load_wav(path)
            if rate is None:
                rate = sr
            elif rate != sr:
                print("Error: Sampling rates must match.", file=sys.stderr)
                sys.exit(1)
            signals.append(audio)
        except RuntimeError as e:
            print(f"Error: {e}", file=sys.stderr)
            sys.exit(1)

    signals = equalize_audio_size(signals)

    # --- Dispatch ---
    if n_mics == 2:
        run_2mic(signals, rate, args)
    else:
        if not args.mic_positions:
            print("Error: --mic-positions is required for >= 3 microphones.", file=sys.stderr)
            sys.exit(1)
        try:
            mic_positions = np.array(json.loads(args.mic_positions), dtype=float)
        except (json.JSONDecodeError, ValueError) as e:
            print(f"Error: invalid --mic-positions JSON: {e}", file=sys.stderr)
            sys.exit(1)
        if mic_positions.ndim != 2 or mic_positions.shape[1] != 3:
            print("Error: --mic-positions must be an Nx3 array of [x,y,z] coordinates.", file=sys.stderr)
            sys.exit(1)
        if mic_positions.shape[0] != n_mics:
            print(f"Error: got {n_mics} WAV files but {mic_positions.shape[0]} mic positions.", file=sys.stderr)
            sys.exit(1)
        run_multimicmic(signals, rate, mic_positions, args)


if __name__ == "__main__":
    main()
