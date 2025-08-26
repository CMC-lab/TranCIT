"""
Simulation utilities.

This module provides utility functions for simulations,
including wavelet generation and other helper functions.
"""

import numpy as np


def morlet(start: float, end: float, num_points: int) -> np.ndarray:
    """
    Generate a Morlet wavelet.

    Args:
    start: Start frequency for the Morlet wavelet.
    end: End frequency for the Morlet wavelet.
    num_points: Length of the perturbation.

    Returns:
    Morlet wavelet.
    """
    t = np.linspace(start, end, num_points)
    w0 = 5
    # sigma = (end - start) / (2 * np.pi)
    sigma = 1
    # sigma = end / (2 * np.pi)  # Standard deviation for Gaussian
    wavelet = (
        (1 / np.sqrt(sigma)) * np.exp(1j * w0 * t) * np.exp(-(t**2) / (2 * (sigma**2)))
    )
    # wavelet = np.exp(1j * start * t) * np.exp(-t ** 2 / (2 * (end ** 2))) # For perturbation
    return np.real(wavelet)
