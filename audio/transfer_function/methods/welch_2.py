import numpy as np
import scipy.signal as signal
import scipy.io.wavfile as wavfile
from scipy.fft import rfft, rfftfreq, irfft
from scipy.fftpack import next_fast_len
import matplotlib.pyplot as plt

def run(reference, acquired, sr):
    print("Starting Transfer Function Estimation (Welch Method)...")

    # 1. Estimate Spectral Densities
    # nperseg defines the length of the impulse response we can recover.
    # 4096 samples @ SR (e.g., 48kHz) is approx 85ms of impulse response.
    nperseg = 4096
    windowing = "hann"
    scaling = "spectrum"

    # Compute Sxx (Auto-spectrum of Input) and Sxy (Cross-spectrum)
    F_xx, S_xx = signal.welch(
        reference, fs=sr, scaling=scaling, window=windowing, nperseg=nperseg
    )
    F_xy, S_xy = signal.csd(
        reference, acquired, fs=sr, scaling=scaling, window=windowing, nperseg=nperseg
    )

    # 2. Calculate Transfer Function H(f)
    epsilon = 1e-10
    H_estimated = S_xy / (S_xx + epsilon)
    H_magnitude = np.abs(H_estimated)
    H_phase = np.angle(H_estimated)

    # Check validity
    if np.any(np.isnan(H_estimated)):
        print("Warning: NaN values in transfer function")
    if np.any(np.isinf(H_estimated)):
        print("Warning: Infinite values in transfer function")

    print(f"H_estimated shape: {H_estimated.shape}")

    # ---------------------------------------------------------
    # CRITICAL FIX: Inverse FFT Handling
    # ---------------------------------------------------------

    # 3. Convert H(f) to Time Domain Impulse Response
    # Note: The length of h_temporal is strictly defined by nperseg.
    # It is roughly nperseg long.
    h_temporal = np.fft.irfft(H_estimated, n=nperseg)

    print(f"Impulse Response length: {len(h_temporal)} samples")

    # 4. Reconstruct Signal via Convolution
    # We do NOT pad h_temporal to match the signal length here.
    # We simply convolve the long reference signal with the short impulse response.
    prediction = signal.fftconvolve(reference, h_temporal, mode='same')

    # 5. Normalize (Gain Matching)
    # The magnitude of H(f) is accurate, but the absolute gain might differ
    # slightly due to windowing/scaling factors. We normalize to match energy.
    scale_factor = np.sum(acquired**2) / (np.sum(prediction**2) + 1e-12)
    # Use sqrt for amplitude scaling
    prediction = prediction * np.sqrt(scale_factor)

    # 6. Calculate Metrics
    mse = np.mean((prediction - acquired) ** 2)
    # Calculate SNR carefully to avoid log(0)
    signal_power = np.var(acquired)
    noise_power = mse
    if noise_power == 0:
        snr = 100.0
    else:
        snr = 10 * np.log10(signal_power / noise_power)

    print(f"MSE: {mse:.6f}")
    print(f"SNR: {snr:.2f} dB")

    # ---------------------------------------------------------
    # Visualization
    # ---------------------------------------------------------
    plt.figure(figsize=(12, 10))

    # Plot 1: Frequency Response
    plt.subplot(3, 1, 1)
    plt.semilogx(F_xx, 20 * np.log10(H_magnitude + 1e-12))
    plt.title('Estimated Transfer Function (Magnitude)')
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Magnitude (dB)')
    plt.grid(True, which='both', linestyle='--', alpha=0.5)
    plt.xlim([20, sr/2])

    # Plot 2: Impulse Response
    plt.subplot(3, 1, 2)
    plt.plot(np.arange(len(h_temporal))/sr, h_temporal)
    plt.title(f'Estimated Impulse Response (Length: {len(h_temporal)/sr:.3f}s)')
    plt.xlabel('Time (s)')
    plt.grid(True)

    # Plot 3: Time Domain Verification (Zoomed)
    plt.subplot(3, 1, 3)
    zoom_len = 5000
    start_idx = len(reference) // 2

    plt.plot(acquired[start_idx:start_idx+zoom_len], label='Acquired', alpha=0.7)
    plt.plot(prediction[start_idx:start_idx+zoom_len], label='Reconstructed (Welch)', linewidth=2, linestyle='--')
    plt.title(f'Verification (SNR: {snr:.2f} dB)')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.show()

    return H_estimated