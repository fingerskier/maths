"""
FFT and Spectral Methods
========================

Implement the Discrete Fourier Transform, explore the FFT algorithm, and apply
spectral methods to differentiation, frequency analysis, and convolution.

Tasks
-----
1. DFT from Scratch: Implement the Discrete Fourier Transform directly from the
   definition: X[k] = sum_{n=0}^{N-1} x[n] * exp(-2*pi*i*k*n/N). Verify
   against numpy.fft.fft.

2. FFT Comparison: Compare your O(N^2) DFT with numpy's FFT on signals of
   increasing length. Measure and plot the wall-clock time of each.

3. Spectral Differentiation: Given a periodic function sampled on [0, 2*pi),
   compute its derivative in the frequency domain by multiplying Fourier
   coefficients by i*k. Compare with the analytical derivative.

4. Frequency Analysis: Generate a signal composed of multiple sinusoids plus
   noise. Use the FFT to identify the constituent frequencies. Return the
   detected frequencies and their amplitudes.

5. Convolution via FFT: Implement linear convolution of two signals using the
   FFT (via the convolution theorem). Compare with direct (naive) convolution.
"""

import numpy as np
import matplotlib.pyplot as plt


def dft(x):
    """
    Compute the Discrete Fourier Transform of a 1D signal.

    Parameters
    ----------
    x : array_like
        Input signal of length N.

    Returns
    -------
    X : np.ndarray
        DFT coefficients (complex), length N.
    """
    raise NotImplementedError


def idft(X):
    """
    Compute the inverse Discrete Fourier Transform.

    Parameters
    ----------
    X : array_like
        Frequency-domain coefficients of length N.

    Returns
    -------
    x : np.ndarray
        Reconstructed time-domain signal (complex), length N.
    """
    raise NotImplementedError


def fft_timing_comparison(sizes=None):
    """
    Compare wall-clock times of custom DFT vs numpy FFT.

    Parameters
    ----------
    sizes : list of int or None
        Signal lengths to test. Defaults to powers of 2 from 2^4 to 2^12.

    Returns
    -------
    sizes : list of int
        Tested sizes.
    dft_times : list of float
        Time in seconds for custom DFT at each size.
    fft_times : list of float
        Time in seconds for numpy FFT at each size.
    """
    raise NotImplementedError


def spectral_derivative(f_values, dx):
    """
    Compute the derivative of a periodic function using spectral differentiation.

    Parameters
    ----------
    f_values : np.ndarray
        Function values sampled uniformly on [0, 2*pi) with spacing dx.
    dx : float
        Grid spacing (2*pi / N).

    Returns
    -------
    df_values : np.ndarray
        Approximate derivative values at the sample points.
    """
    raise NotImplementedError


def frequency_analysis(signal, sample_rate):
    """
    Identify dominant frequencies in a signal using the FFT.

    Parameters
    ----------
    signal : np.ndarray
        Time-domain signal.
    sample_rate : float
        Sampling frequency in Hz.

    Returns
    -------
    frequencies : np.ndarray
        Detected dominant frequencies (in Hz).
    amplitudes : np.ndarray
        Corresponding amplitudes.
    """
    raise NotImplementedError


def fft_convolution(a, b):
    """
    Compute the linear convolution of two signals using the FFT.

    Parameters
    ----------
    a, b : array_like
        Input signals.

    Returns
    -------
    result : np.ndarray
        Convolution result of length len(a) + len(b) - 1.
    """
    raise NotImplementedError


if __name__ == "__main__":
    print("=== DFT Verification ===")
    x = np.random.randn(64)
    X_custom = dft(x)
    X_numpy = np.fft.fft(x)
    print(f"Max difference from numpy: {np.max(np.abs(X_custom - X_numpy)):.2e}")

    print("\n=== IDFT Verification ===")
    x_recon = idft(X_custom)
    print(f"Reconstruction error: {np.max(np.abs(x_recon - x)):.2e}")

    print("\n=== FFT Timing Comparison ===")
    sizes, dft_t, fft_t = fft_timing_comparison()
    for s, d, f in zip(sizes, dft_t, fft_t):
        print(f"N={s:5d}: DFT={d:.4f}s, FFT={f:.6f}s, ratio={d/f:.1f}x")

    print("\n=== Spectral Differentiation ===")
    N = 128
    dx = 2 * np.pi / N
    x_grid = np.arange(N) * dx
    f_vals = np.sin(x_grid)
    df_vals = spectral_derivative(f_vals, dx)
    exact_df = np.cos(x_grid)
    print(f"Max derivative error: {np.max(np.abs(df_vals - exact_df)):.2e}")

    print("\n=== Frequency Analysis ===")
    t = np.linspace(0, 1, 1000, endpoint=False)
    signal = 3 * np.sin(2 * np.pi * 50 * t) + np.sin(2 * np.pi * 120 * t)
    freqs, amps = frequency_analysis(signal, sample_rate=1000)
    print(f"Detected frequencies: {freqs}")
    print(f"Amplitudes: {amps}")

    print("\n=== FFT Convolution ===")
    a = np.array([1, 2, 3])
    b = np.array([0, 1, 0.5])
    conv_fft = fft_convolution(a, b)
    conv_np = np.convolve(a, b)
    print(f"FFT convolution error: {np.max(np.abs(conv_fft - conv_np)):.2e}")
