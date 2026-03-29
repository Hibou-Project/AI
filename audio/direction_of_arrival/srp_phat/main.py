from __future__ import annotations

import argparse
import sys
from pathlib import Path

if __package__:
    from .audio import load_signals
    from .config import MIC_DIRECTIONS, MIC_POSITIONS, SAMPLE_RATE
    from .peak import find_top_k_peaks
    from .report import plot_srp_map, print_report
    from .simulation import save_simulation_wavs, simulate_signals
    from .srp_engine import SRPEngine
else:
    # Support direct execution: `python direction_of_arrival/srp_phat/main.py`
    repo_root = Path(__file__).resolve().parents[2]
    root_str = str(repo_root)
    if root_str not in sys.path:
        sys.path.insert(0, root_str)
    from direction_of_arrival.srp_phat.audio import load_signals
    from direction_of_arrival.srp_phat.config import (
        MIC_DIRECTIONS,
        MIC_POSITIONS,
        SAMPLE_RATE,
    )
    from direction_of_arrival.srp_phat.peak import find_top_k_peaks
    from direction_of_arrival.srp_phat.report import plot_srp_map, print_report
    from direction_of_arrival.srp_phat.simulation import (
        save_simulation_wavs,
        simulate_signals,
    )
    from direction_of_arrival.srp_phat.srp_engine import SRPEngine


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Modular SRP-PHAT drone DOA (16 kHz)")
    parser.add_argument(
        "wav_files",
        nargs="*",
        default=["1.wav", "2.wav"],
        help="WAV file per mic (default: 1.wav 2.wav)",
    )
    parser.add_argument(
        "--target-fs",
        type=int,
        default=SAMPLE_RATE,
        help=f"Processing sample rate (default: {SAMPLE_RATE})",
    )
    parser.add_argument("--plot", action="store_true", help="Show and save SRP map")
    parser.add_argument(
        "--simulate",
        type=float,
        default=None,
        metavar="DEG",
        help="Simulate source at DEG instead of loading WAV files",
    )
    parser.add_argument("--snr", type=float, default=20.0, help="Simulation SNR (dB)")
    parser.add_argument("--top", type=int, default=3, help="Top candidate directions")
    parser.add_argument(
        "--save-sim",
        action="store_true",
        help="Save simulated channels as WAV in ./simulation_signals/",
    )
    return parser


def main() -> float:
    args = build_parser().parse_args()

    true_angle = None
    if args.simulate is not None:
        true_angle = args.simulate % 360.0
        fs = args.target_fs
        signals = simulate_signals(
            source_angle_deg=true_angle,
            fs=fs,
            snr_db=args.snr,
            mic_positions=MIC_POSITIONS,
        )
        if args.save_sim:
            written = save_simulation_wavs(signals, fs, true_angle)
            for path in written:
                print(f"  wrote: {path}")
    else:
        if len(args.wav_files) != len(MIC_POSITIONS):
            print(
                f"ERROR: expected {len(MIC_POSITIONS)} WAV files "
                f"(one per mic), got {len(args.wav_files)}"
            )
            sys.exit(1)
        signals, fs = load_signals(args.wav_files, target_fs=args.target_fs)

    engine = SRPEngine(fs=fs, mic_positions=MIC_POSITIONS, mic_directions=MIC_DIRECTIONS)
    srp_map = engine.process(signals)
    estimated_doa, confidence = engine.find_doa(srp_map)
    peaks = find_top_k_peaks(srp_map, engine.theta_grid, k=args.top)
    print_report(estimated_doa, confidence, peaks)

    if true_angle is not None:
        error = abs((estimated_doa - true_angle + 180.0) % 360.0 - 180.0)
        print(f"  simulation error: {error:.2f} deg")

    if args.plot or args.simulate is not None:
        plot_srp_map(
            srp_map=srp_map,
            theta_grid=engine.theta_grid,
            estimated_doa=estimated_doa,
            mic_directions=MIC_DIRECTIONS,
            true_angle=true_angle,
            save_path="srp_map.png",
        )

    return estimated_doa


if __name__ == "__main__":
    main()
