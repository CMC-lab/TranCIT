from typing import Dict, Tuple

import numpy as np


def estimate_residuals(event_stats: Dict) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Estimate residuals and related statistics for each time step in a VAR model.

    Parameters
    ----------
    event_stats : Dict
        Dictionary containing:
        - 'OLS': Sub-dictionary with 'At' (coefficients), shape (L, nvar, nvar * morder).
        - 'Sigma': Covariance matrices, shape (L, nvar * (morder + 1), nvar * (morder + 1)).
        - 'mean': Mean values, shape (nvar * (morder + 1), L).

    Returns
    -------
    Tuple[np.ndarray, np.ndarray, np.ndarray]
        - residual_biases : Shape (nvar, L), residual biases.
        - residual_covariance : Shape (L, nvar, nvar), residual covariance matrices.
        - residual_trace : Shape (L, 1), trace of residual covariance matrices.
    """
    L, nvar, temp = event_stats["OLS"]["At"].shape
    residual_biases = np.full((nvar, L), np.nan)
    residual_covariance = np.full((L, nvar, nvar), np.nan)
    residual_trace = np.full((L, 1), np.nan)

    for t in range(L):
        Sigma_Xt = event_stats["Sigma"][t, :nvar, :nvar]
        Sigma_Xp = event_stats["Sigma"][t, nvar:, nvar:]
        Sigma_XtXp = event_stats["Sigma"][t, :nvar, nvar:].reshape(nvar, temp)
        coeff = event_stats["OLS"]["At"][t].reshape(nvar, temp)

        residual_biases[:, t] = event_stats["mean"][:nvar, t] - np.dot(
            coeff, event_stats["mean"][nvar:, t]
        )
        residual_covariance[t] = (
            Sigma_Xt
            - np.dot(Sigma_XtXp, coeff.T)
            - np.dot(coeff, Sigma_XtXp.T)
            + np.dot(np.dot(coeff, Sigma_Xp), coeff.T)
        )
        residual_trace[t] = np.trace(residual_covariance[t])

    # Ensure that no negative values exist (if needed)
    # Sigma_Et[Sigma_Et < 0] = 0
    # sigma_Et[sigma_Et < 0] = 0

    return residual_biases, residual_covariance, residual_trace


def get_residuals(event_data: np.ndarray, stats: Dict) -> np.ndarray:
    """
    Calculate residuals for each time step in a VAR model.

    Parameters
    ----------
    event_data : np.ndarray
        Event matrix, shape (nvar * (morder + 1), L, ntrials).
    stats : Dict
        Dictionary containing:
        - 'OLS': Sub-dictionary with 'At' (coefficients), shape (L, nvar, nvar * morder).

    Returns
    -------
    np.ndarray
        Residuals, shape (nvar, L, ntrials).

    Raises
    ------
    ValueError
        If coefficient and lagged data shapes are incompatible.
    """
    L, nvar, _ = stats["OLS"]["At"].shape
    ntrials = event_data.shape[2]
    residuals = np.full((nvar, L, ntrials), np.nan)

    for t in range(L):
        Xt = event_data[:nvar, t, :]
        Xp = event_data[nvar:, t, :]
        coeff = stats["OLS"]["At"][t]

        if coeff.shape[1] != Xp.shape[0]:
            raise ValueError(
                f"Shape mismatch at time {t}: coeff.shape[1]={coeff.shape[1]}, "
                f"Xp.shape[0]={Xp.shape[0]}"
            )

        residuals[:, t, :] = Xt - np.dot(coeff, Xp)

    return residuals
