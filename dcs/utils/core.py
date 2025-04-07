import logging
from typing import Dict

import numpy as np

from .preprocess import regularize_if_singular
from .residuals import estimate_residuals

logging.basicConfig(level=logging.INFO)


def extract_event_windows(
    signal: np.ndarray, centers: np.ndarray, start_offset: int, window_length: int
) -> np.ndarray:
    """
    Extract windows of data from a signal around specified center points.

    Parameters
    ----------
    signal : np.ndarray
        1D array representing the signal data.
    centers : np.ndarray
        1D array of center points (indices) around which to extract windows.
    start_offset : int
        Offset from the center to start the window.
    window_length : int
        Length of each window to extract.

    Returns
    -------
    np.ndarray
        2D array of shape (window_length, len(centers)) containing the extracted windows.

    Raises
    ------
    IndexError
        If the calculated indices for any window are out of bounds for the signal array.
    """
    event_windows = np.full((window_length, len(centers)), np.nan)

    for i, center in enumerate(centers):
        start_idx = int(np.round(center - start_offset))
        end_idx = start_idx + window_length
        idx = np.arange(start_idx, end_idx)

        if np.any(idx < 0) or np.any(idx >= len(signal)):
            logging.error(
                f"Index out of bounds for center {center}: {idx} for signal of length {len(signal)}"
            )
            raise IndexError(
                f"Index out of bounds: {idx} for array of length {len(signal)}"
            )

        event_windows[:, i] = signal[idx]

    return event_windows


def compute_conditional_event_statistics(
    event_data: np.ndarray, model_order: int, epsilon: float = 1e-4
) -> Dict:
    """
    Compute conditional statistics for VAR time series events, including mean and covariance.

    Parameters
    ----------
    event_data : np.ndarray
        VAR time series events of shape (nvar * (model_order + 1), time points, trials).
    model_order : int
        The model order for the VAR process.
    epsilon : float, optional
        Small value for regularization if the matrix is singular. Default is 1e-4.

    Returns
    -------
    dict
        Dictionary containing the conditional statistics:
            - 'mean': Mean of the events (shape: (nvar * (model_order + 1), time points)).
            - 'Sigma': Covariance matrices (shape: (time points, nvar * (model_order + 1), nvar * (model_order + 1))).
            - 'OLS': Dictionary with:
                - 'At': OLS coefficients (shape: (time points, nvar, nvar * model_order)).
                - 'bt': Residual biases.
                - 'Sigma_Et': Residual covariance.
                - 'sigma_Et': Residual standard deviation.
    """
    nvar = event_data.shape[0] // (model_order + 1)
    stats = {
        "mean": np.mean(event_data, axis=2),
        "Sigma": np.zeros(
            (event_data.shape[1], nvar * (model_order + 1), nvar * (model_order + 1))
        ),
        "OLS": {"At": np.zeros((event_data.shape[1], nvar, nvar * model_order))},
    }

    for t in range(event_data.shape[1]):
        temp = event_data[:, t, :] - stats["mean"][:, t, np.newaxis]
        stats["Sigma"][t, :, :] = np.dot(temp, temp.T) / event_data.shape[2]

        Sigma_sub_matrix = stats["Sigma"][t, :nvar, nvar:]
        Sigma_end = stats["Sigma"][t, nvar:, nvar:]

        if np.linalg.det(Sigma_end) == 0:
            logging.warning(
                f"Matrix singular at time {t}, applying regularization with epsilon={epsilon}"
            )
            Sigma_end = regularize_if_singular(Sigma_end, epsilon)

        stats["OLS"]["At"][t, :, :] = np.dot(Sigma_sub_matrix, np.linalg.inv(Sigma_end))

    stats["OLS"]["bt"], stats["OLS"]["Sigma_Et"], stats["OLS"]["sigma_Et"] = (
        estimate_residuals(stats)
    )
    return stats


def extract_event_snapshots(
    time_series: np.ndarray,
    locations: np.ndarray,
    model_order: int,
    lag_step: int,
    start_offset: int,
    extract_length: int,
) -> np.ndarray:
    """
    Extract event snapshots from time series data for multiple variables and lags.

    Parameters
    ----------
    time_series : np.ndarray
        2D array of shape (variables, time points) containing the time series data.
    locations : np.ndarray
        1D array of event locations (indices).
    model_order : int
        The model order (number of lags).
    lag_step : int
        The step size for lags.
    start_offset : int
        Offset from the location to start the window.
    extract_length : int
        Length of each extracted window.

    Returns
    -------
    np.ndarray
        3D array of shape (variables * (model_order + 1), extract_length, len(locations))
        containing the extracted event snapshots.
    """
    nvar = time_series.shape[0]
    snapshots = np.full(
        (nvar * (model_order + 1), extract_length, len(locations)), np.nan
    )

    idx1 = np.arange(nvar * (model_order + 1))
    idx2 = np.tile(np.arange(nvar), model_order + 1)
    delay = np.tile(np.arange(0, model_order + 1) * lag_step, (nvar, 1)).flatten()

    for n in range(len(idx1)):
        snapshots[idx1[n], :, :] = extract_event_windows(
            time_series[idx2[n], :], locations - delay[n], start_offset, extract_length
        )

    return snapshots


def compute_event_statistics(event_data: np.ndarray, model_order: int) -> Dict:
    """
    Compute statistics for VAR time series events, including mean, covariance, and OLS coefficients.

    Parameters
    ----------
    event_data : np.ndarray
        VAR time series events of shape (variables * (model_order + 1), time points, trials).
    model_order : int
        The model order for the VAR process.

    Returns
    -------
    dict
        Dictionary containing the event statistics:
            - 'mean': Mean of the events (shape: (variables * (model_order + 1), time points)).
            - 'n_trials': Number of trials.
            - 'Sigma': Covariance matrices (shape: (time points, nvar * (model_order + 1), nvar * (model_order + 1))).
            - 'OLS': Dictionary with:
                - 'At': OLS coefficients (shape: (time points, nvar, nvar * model_order)).
                - 'bt': Residual biases.
                - 'Sigma_Et': Residual covariance.
                - 'sigma_Et': Residual standard deviation.
    """
    nvar = event_data.shape[0] // (model_order + 1)
    nobs = event_data.shape[1]
    n_trials = event_data.shape[2]

    stats = {
        "mean": np.mean(event_data, axis=2),
        "n_trials": n_trials,
        "Sigma": np.zeros((nobs, nvar * (model_order + 1), nvar * (model_order + 1))),
        "OLS": {"At": np.zeros((nobs, nvar, nvar * model_order))},
    }

    for t in range(nobs):
        temp = event_data[:, t, :] - stats["mean"][:, t][:, np.newaxis]
        stats["Sigma"][t, :, :] = np.dot(temp, temp.T) / n_trials

        Sigma_12 = stats["Sigma"][t, :nvar, nvar:]  # Shape: (nvar, nvar * model_order)
        Sigma_22 = stats["Sigma"][
            t, nvar:, nvar:
        ]  # Shape: (nvar * model_order, nvar * model_order)

        Sigma_22_reg = regularize_if_singular(Sigma_22)
        if not np.allclose(Sigma_22, Sigma_22_reg):
            logging.warning(f"Applied regularization to Sigma_22 at time step {t}")

        stats["OLS"]["At"][t, :, :] = Sigma_12 @ np.linalg.inv(Sigma_22_reg)

    stats["OLS"]["bt"], stats["OLS"]["Sigma_Et"], stats["OLS"]["sigma_Et"] = (
        estimate_residuals(stats)
    )
    return stats
