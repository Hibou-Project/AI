import numpy as np
import matplotlib.pyplot as plt


def display_over_time(reference, acquired, sr):
    time_ref = np.arange(len(reference)) / sr
    time_acq = np.arange(len(acquired)) / sr

    # reference_norm = (
    #     reference / np.max(np.abs(reference))
    #     if np.max(np.abs(reference)) > 0
    #     else reference
    # )
    # acquired_norm = (
    #     acquired / np.max(np.abs(acquired))
    #     if np.max(np.abs(acquired)) > 0
    #     else acquired
    # )
    reference_norm = reference
    acquired_norm = acquired

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))

    # Plot reference signal
    ax1.plot(time_ref, reference_norm, label="Reference", alpha=0.7, linewidth=0.5)
    ax1.set_xlabel("Time (s)")
    ax1.set_ylabel("Amplitude (normalized)")
    ax1.set_title("Reference Signal")
    ax1.grid(True, alpha=0.3)
    ax1.legend()

    # Plot acquired signal
    ax2.plot(
        time_acq,
        acquired_norm,
        label="Acquired",
        alpha=0.7,
        linewidth=0.5,
        color="orange",
    )
    ax2.set_xlabel("Time (s)")
    ax2.set_ylabel("Amplitude (normalized)")
    ax2.set_title("Acquired Signal")
    ax2.grid(True, alpha=0.3)
    ax2.legend()

    plt.tight_layout()
    plt.savefig("img/signals_over_time.png")

    # Print signal information
    print(
        f"Reference signal: {len(reference)} samples, duration: {len(reference)/sr:.3f} s"
    )
    print(
        f"Acquired signal: {len(acquired)} samples, duration: {len(acquired)/sr:.3f} s"
    )
    print(f"Sample rate: {sr} Hz")


import numpy as np
from scipy.signal import find_peaks


def remove_pulses(signal_slice, threshold_mult=5, window_radius=200, min_distance=500):
    """
    Nettoie une tranche de signal en remplaçant les pics (positifs et négatifs)
    par une interpolation linéaire.

    Parameters:
    -----------
    signal_slice : np.array
        Le signal audio à nettoyer.
    threshold_mult : float
        Multiplicateur de l'écart-type pour définir le seuil de détection (défaut: 5).
    window_radius : int
        Rayon (en échantillons) autour du pic à effacer (défaut: 200).
    min_distance : int
        Distance minimale entre deux pics pour éviter les doublons (défaut: 500).

    Returns:
    --------
    np.array
        Le signal nettoyé.
    """

    # On travaille sur une copie pour ne pas modifier l'original par inadvertance
    s = signal_slice.astype(np.float64)

    # Calcul du seuil dynamique basé sur le bruit du signal
    mean_val = np.mean(s)
    std_val = np.std(s)
    threshold = mean_val + threshold_mult * std_val

    # Détection des pics positifs
    pos_peaks, _ = find_peaks(s, height=threshold, distance=min_distance)

    # Détection des pics négatifs (on inverse le signal)
    neg_peaks, _ = find_peaks(-s, height=threshold, distance=min_distance)

    # Fusion des indices
    all_peaks = np.concatenate((pos_peaks, neg_peaks))

    # Si aucun pic n'est détecté, on retourne le signal tel quel
    if len(all_peaks) == 0:
        return s

    # Création du masque (False = garder, True = remplacer)
    mask = np.zeros(len(s), dtype=bool)

    for p in all_peaks:
        start = max(0, p - window_radius)
        end = min(len(s), p + window_radius)
        mask[start:end] = True

    # Interpolation pour "boucher" les trous
    s_clean = s.copy()
    s_clean[mask] = np.nan  # On utilise NaN comme marqueurs pour l'interpolation

    x = np.arange(len(s))
    # np.interp ne garde que les valeurs valides (~mask) pour tracer la ligne
    s_clean = np.interp(x, x[~mask], s_clean[~mask])

    return s_clean


def remove_pulses_robust(
    signal_slice, prominence_val=None, window_radius=200, min_distance=500
):
    """
    Version robuste utilisant la 'prominence' pour capter les pics et vallées
    même si le signal n'est pas centré sur 0.
    """
    s = signal_slice.astype(np.float64)

    # Estimation automatique de la saillance si non fournie
    # On utilise 5 fois l'écart-type comme base, comme avant.
    if prominence_val is None:
        prominence_val = 5 * np.std(s)

    # 1. Détection par PROMINENCE (Saillance)
    # On détecte ce qui dépasse localement, peu importe le niveau global
    pos_peaks, _ = find_peaks(s, prominence=prominence_val, distance=min_distance)
    neg_peaks, _ = find_peaks(-s, prominence=prominence_val, distance=min_distance)

    # Fusion des indices
    all_peaks = np.concatenate((pos_peaks, neg_peaks))

    if len(all_peaks) == 0:
        return s

    # MAsk creation
    mask = np.zeros(len(s), dtype=bool)
    for p in all_peaks:
        start = max(0, p - window_radius)
        end = min(len(s), p + window_radius)
        mask[start:end] = True

    # Interpolation
    s_clean = s.copy()
    s_clean[mask] = np.nan

    x = np.arange(len(s))
    s_clean = np.interp(x, x[~mask], s_clean[~mask])

    return s_clean
