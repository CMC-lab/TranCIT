"""
Coupled oscillator simulation functions.

This module provides functions for simulating coupled oscillator systems,
commonly used in neuroscience and physics applications.
"""

from typing import Tuple

import numpy as np

from .utils import morlet


def generate_signals(
    T: int,
    Ntrial: int,
    h: float,
    gamma1: float,
    gamma2: float,
    Omega1: float,
    Omega2: float,
    apply_morlet: bool = False,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Generate bivariate coupled oscillator signals based on a specified model.

    Simulates two coupled second-order linear differential equations discretized
    using a time step h, with added noise terms. Allows for optional non-stationarity
    in the noise applied to the first signal (x) via a Morlet wavelet shape.

    Parameters
    ----------
    T : int
        Total number of time points to simulate (including initial points).
    Ntrial : int
        Number of trials (realizations) to generate.
    h : float
        Time step for discretization.
    gamma1 : float
        Damping coefficient for the first oscillator (x).
    gamma2 : float
        Damping coefficient for the second oscillator (y).
    Omega1 : float
        Natural frequency for the first oscillator (x).
    Omega2 : float
        Natural frequency for the second oscillator (y).
    apply_morlet : bool, optional
        If True, applies a Morlet wavelet shape to modulate the noise variance (`ns_x`)
        for the first signal, introducing non-stationarity. Defaults to False, using
        constant noise variance.

    Returns
    -------
    Tuple[np.ndarray, np.ndarray, np.ndarray]
        - X : Generated signals, shape (2, T - burnin, Ntrial). Contains x and y
          signals after discarding burn-in points.
        - ns_x : Noise variance profile for the x signal, shape (T + 1,).
        - ns_y : Noise variance profile for the y signal, shape (T + 1,).
    """
    burnin = min(500, T // 4)  # Use smaller burnin for short series
    X = np.zeros((2, T - burnin, Ntrial))

    if apply_morlet is True:
        ns_x = 0.02 * np.concatenate(
            [np.ones(650), np.ones(201) - morlet(-0.29, 0.29, 201), np.ones(150)]
        )
    else:
        ns_x = 0.02 * np.ones(T + 1)

    ns_y = 0.005 * np.ones(T + 1)

    for N in range(Ntrial):
        x = np.random.rand(2)
        y = np.random.rand(2)

        c2 = 0
        c1 = 0.098

        for t in range(1, T - 1):
            x = np.append(
                x,
                (2 - gamma1 * h) * x[-1]
                + (-1 + gamma1 * h - h**2 * Omega1**2) * x[-2]
                + h**2 * ns_x[t] * np.random.randn()
                + h**2 * c2 * y[-2],
            )
            y = np.append(
                y,
                (2 - gamma2 * h) * y[-1]
                + (-1 + gamma2 * h - h**2 * Omega2**2) * y[-2]
                + h**2 * ns_y[t] * np.random.randn()
                + h**2 * c1 * x[-2],
            )

        u = np.array([x[burnin:], y[burnin:]])
        X[:, :, N] = u

    return X, ns_x, ns_y
