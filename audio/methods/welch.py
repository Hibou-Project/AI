import numpy as np
import scipy.signal as signal
import scipy.io.wavfile as wavfile
from scipy.fft import rfft, rfftfreq, irfft
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
from scipy.signal import welch, windows
from scipy import interpolate
from scipy.fftpack import next_fast_len
import threading

"""
H(wt)=Y(wt)/X(wt), Y=fft(x), X=fft(y)
1. Compute cross spectral density Sxy (DSP): using Welch method and Hann windows
2. Compute Sxx (SP): spectral density
Estimating H: Ĥ=Sxy/Sxx
"""


def run(reference, acquired, sr):
    nperseg = 4096
    windowing = "hann"
    scaling = "spectrum"
    F_xx, S_xx = welch(
        reference, fs=sr, scaling=scaling, window=windowing, nperseg=nperseg
    )
    F_xy, S_xy = signal.csd(
        reference, acquired, fs=sr, scaling=scaling, window=windowing, nperseg=nperseg
    )

    ref_var = np.var(reference)
    acq_var = np.var(acquired)
    hann_var = np.sum(S_xx)

    print("Reference signal variance:", ref_var)
    print("Acquired signal variance:", acq_var)
    print("Hann variance for XX:", hann_var)
    print("Ratio:", ref_var / hann_var)

    epsilon = 1e-10
    H_estimated = S_xy / (S_xx + epsilon)
    H_magnitude = np.abs(H_estimated)
    H_phase = np.angle(H_estimated)

    # Add these after computing H_estimated
    if np.any(np.isnan(H_estimated)):
        print("Warning: NaN values in transfer function")
    if np.any(np.isinf(H_estimated)):
        print("Warning: Infinite values in transfer function (divide by zero?)")

    print("Estimated Transfer Function computed.")
    print("Shape of H:", H_estimated.shape)
    print("Frequencies shape:", F_xx.shape)
    print("Frequencies range:", F_xx[0], "to", F_xx[-1], "Hz")
    print("First 5 Magnitudes:", H_magnitude[:5])
    print("First 5 Phases:", H_phase[:5])

    # Need to perform inverse FFT to get temporal response with tricks or it will take ages.

    # 1. Pad to optimal length (2*3*5 factors)
    n_opt = next_fast_len(len(reference))

    H_padded = np.pad(H_estimated, (0, n_opt - len(H_estimated)))
    H_temporal = np.fft.irfft(H_padded, n=n_opt)[: len(reference)]  # Trim back
    print("Temporal response length:", len(H_temporal))

    # Or via frequency domain multiplication
    n_fft = next_fast_len(len(reference) + len(H_temporal) - 1)
    ref_fft = np.fft.rfft(reference, n=n_fft)
    h_fft = np.fft.rfft(H_temporal, n=n_fft)
    prediction_freq = ref_fft * h_fft
    prediction = np.fft.irfft(prediction_freq, n=n_fft)[: len(acquired)]

    # Calculate metrics
    mse = np.mean((prediction - acquired) ** 2)
    snr = 10 * np.log10(np.var(acquired) / mse)
    print(f"MSE: {mse:.6f}, SNR: {snr:.2f} dB")

    # Caluclate H_temporal(reference) to get predicted signal
    # pred_var = np.var(prediction)

    # print("Predicted signal variance:", pred_var)
    # print("Match ratio:", pred_var / acq_var)
