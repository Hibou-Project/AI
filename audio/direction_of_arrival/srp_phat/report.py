from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np


def compass_label(deg: float) -> str:
    labels = [
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
    return labels[int((deg + 11.25) / 22.5) % 16]


def print_report(
    estimated_doa: float,
    confidence: float,
    peaks: list[tuple[float, float]],
) -> None:
    print("\n" + "=" * 56)
    print("  DRONE DIRECTION OF ARRIVAL — RESULTS")
    print("=" * 56)
    print(f"  Primary DOA : {estimated_doa:6.1f}° ({compass_label(estimated_doa)})")
    print(f"  Confidence  : {confidence:6.2f}x")
    print("\n  Top candidate directions:")
    peak_max = max((p for _, p in peaks), default=1.0) or 1.0
    for rank, (angle, power) in enumerate(peaks, start=1):
        bar = "█" * int(20 * (power / peak_max))
        print(
            f"    #{rank} {angle:6.1f}° ({compass_label(angle):3s})"
            f" power={power:.4f} {bar}"
        )
    print("=" * 56 + "\n")


def plot_srp_map(
    srp_map: np.ndarray,
    theta_grid: np.ndarray,
    estimated_doa: float,
    mic_directions: np.ndarray,
    true_angle: float | None = None,
    save_path: str = "srp_map.png",
) -> None:
    theta_rad = np.radians(theta_grid)
    norm_srp = srp_map / (np.max(np.abs(srp_map)) + 1e-12)
    norm_srp = np.clip(norm_srp, 0.0, None)

    fig = plt.figure(figsize=(12, 5))
    ax_polar = fig.add_subplot(1, 2, 1, projection="polar")
    ax_linear = fig.add_subplot(1, 2, 2)

    ax_polar.plot(theta_rad, norm_srp, color="royalblue")
    ax_polar.fill(theta_rad, norm_srp, color="royalblue", alpha=0.2)
    ax_polar.set_theta_zero_location("E")
    ax_polar.set_theta_direction(1)
    ax_polar.axvline(np.radians(estimated_doa), color="red", linewidth=2)
    if true_angle is not None:
        ax_polar.axvline(np.radians(true_angle), color="green", linestyle="--")
    for idx, mic_dir in enumerate(mic_directions):
        rad = np.radians(mic_dir)
        ax_polar.plot([rad, rad], [0.0, 0.25], color="orange", linewidth=2)
        ax_polar.text(rad, 0.3, f"M{idx + 1}", color="darkorange", ha="center")
    ax_polar.set_title("Polar SRP map")

    ax_linear.plot(theta_grid, norm_srp, color="royalblue")
    ax_linear.fill_between(theta_grid, 0.0, norm_srp, color="royalblue", alpha=0.2)
    ax_linear.axvline(estimated_doa, color="red", label=f"DOA {estimated_doa:.1f}°")
    if true_angle is not None:
        ax_linear.axvline(
            true_angle, color="green", linestyle="--", label=f"True {true_angle:.1f}°"
        )
    ax_linear.set_xlim(0, 360)
    ax_linear.set_xlabel("Direction (deg)")
    ax_linear.set_ylabel("Normalised SRP")
    ax_linear.grid(alpha=0.3)
    ax_linear.legend(fontsize=8)
    ax_linear.set_title("Linear SRP map")

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"  Plot saved -> {save_path}")
    plt.show()
