import argparse
import sys
import wave
import numpy as np
import pyaudio

# --- Helper Functions (Same as before) ---

def play_audio(audio: np.ndarray, sr: int) -> None:
    p = pyaudio.PyAudio()
    stream = p.open(format=pyaudio.paFloat32, channels=1, rate=sr, output=True)
    try:
        stream.write(audio.tobytes())
    finally:
        stream.stop_stream()
        stream.close()
        p.terminate()

def gcc_phat(sig, refsig, fs=1, max_tau=None, interp=16):
    n = sig.shape[0] + refsig.shape[0]
    SIG = np.fft.rfft(sig, n=n)
    REFSIG = np.fft.rfft(refsig, n=n)
    R = SIG * np.conj(REFSIG)
    
    eps = np.finfo(float).eps
    cc = np.fft.irfft(R / (np.abs(R) + eps), n=(interp * n))

    max_shift = int(interp * n / 2)
    if max_tau:
        max_shift = np.minimum(int(interp * fs * max_tau), max_shift)

    cc = np.concatenate((cc[-max_shift:], cc[: max_shift + 1]))
    shift = np.argmax(np.abs(cc)) - max_shift
    tau = shift / float(interp * fs)
    return tau, cc

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

    # Support 16-bit and 24-bit
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
        # Assuming 32-bit float for common recording formats, fallback to int32
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

# --- Main Logic ---

def main() -> None:
    parser = argparse.ArgumentParser(description="GCC-PHAT DoA on Chunks")
    parser.add_argument("wav_files", nargs="+", help="Two WAV files (Mic 1, Mic 2)")
    parser.add_argument("--distance", type=float, default=0.5, help="Distance between mics (meters)")
    parser.add_argument("--chunk", type=float, default=0.5, help="Chunk size in seconds (default: 0.5)")
    parser.add_argument("--debug", action="store_true")
    
    args = parser.parse_args()

    if len(args.wav_files) < 2:
        print("Error: Provide two WAV files.", file=sys.stderr)
        sys.exit(1)

    # 1. Load Audio
    print("Loading files...")
    signals = []
    rate = None
    for path in args.wav_files:
        try:
            audio, sr = load_wav(path)
            if rate is None: rate = sr
            elif rate != sr:
                print("Error: Sampling rates must match.", file=sys.stderr)
                sys.exit(1)
            signals.append(audio)
        except RuntimeError as e:
            print(f"Error: {e}", file=sys.stderr)
            sys.exit(1)

    # Truncate to shortest length
    signals = equalize_audio_size(signals)
    sig1 = signals[0]
    sig2 = signals[1]
    
    # Constants
    c = 343.0
    L = args.distance
    max_tau = L / c
    chunk_samples = int(rate * args.chunk)
    total_samples = len(sig1)
    
    print(f"Processing {total_samples/rate:.2f}s audio in {args.chunk}s chunks...")
    print("-" * 60)
    print(f"{'Time (s)':<10} | {'Angle (deg)':<12} | {'Delay (ms)':<12} | Status")
    print("-" * 60)

    # 2. Iterate over chunks
    # We use a step size of chunk_samples (no overlap). 
    # For smoother tracking, you can change 'step' to be smaller than 'chunk_samples'.
    step = chunk_samples 
    
    for i in range(0, total_samples - chunk_samples, step):
        s1_chunk = sig1[i : i + chunk_samples]
        s2_chunk = sig2[i : i + chunk_samples]
        
        # Simple Energy Threshold to ignore silence
        # If RMS energy is too low, GCC-PHAT returns garbage angles
        energy1 = np.sqrt(np.mean(s1_chunk**2))
        if energy1 < 0.01: # Threshold for silence (tune based on your audio)
            print(f"{i/rate:<10.2f} | {'---':<12} | {'---':<12} | Silent")
            continue

        # 3. Compute GCC-PHAT for this chunk
        tau, _ = gcc_phat(s2_chunk, s1_chunk, fs=rate, max_tau=max_tau)
        
        # 4. Calculate Angle
        delta_d = c * tau
        ratio = delta_d / L
        
        # Clipping for safety
        ratio = np.clip(ratio, -1.0, 1.0)
        theta = np.degrees(np.arcsin(ratio))
        
        print(f"{i/rate:<10.2f} | {theta:<12.2f} | {tau*1000:<12.4f} | OK")

    print("-" * 60)
    print("Done.")

if __name__ == "__main__":
    main()