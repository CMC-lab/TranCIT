import logging
from typing import Tuple

import numpy as np

from .preprocess import regularize_if_singular

logging.basicConfig(level=logging.INFO)

def compute_covariances(lagged_data_array: np.ndarray, n_time_steps: int, model_order: int) -> tuple:
    """
    Compute covariance matrices for lagged data.

    Parameters
    ----------
    lagged_data_array : np.ndarray
        Lagged data array of shape (variables, model_order, time_steps, trials).
        The first dimension corresponds to variables, with index 0 as Y and index 1 as X.
    n_time_steps : int
        Number of time steps.
    model_order : int
        Model order (number of lags).

    Returns
    -------
    tuple
        A tuple containing:
        - cov_Xp : np.ndarray
            Covariance of X past (shape: (time_steps, model_order, model_order)).
        - cov_Yp : np.ndarray
            Covariance of Y past (shape: (time_steps, model_order, model_order)).
        - C_XYp : np.ndarray
            Cross-covariance X past to Y past (shape: (time_steps, model_order, model_order)).
        - C_YXp : np.ndarray
            Cross-covariance Y past to X past (shape: (time_steps, model_order, model_order)).

    Notes
    -----
    This function computes covariance matrices for each time step using NumPy's `np.cov`.
    It assumes two variables (Y and X) and processes their lagged data accordingly.
    """
    # Initialize output arrays
    cov_Xp = np.zeros((n_time_steps, model_order, model_order))
    cov_Yp = np.zeros((n_time_steps, model_order, model_order))
    C_XYp = np.zeros((n_time_steps, model_order, model_order))
    C_YXp = np.zeros((n_time_steps, model_order, model_order))

    # Iterate over time steps
    for t in range(n_time_steps):
        # Extract past data for X and Y, transposed to (trials, model_order)
        X_past = lagged_data_array[1, :, t, :].T
        Y_past = lagged_data_array[0, :, t, :].T

        # Compute covariance matrices
        cov_Xp[t] = np.cov(X_past, rowvar=False)  # Covariance of X past
        cov_Yp[t] = np.cov(Y_past, rowvar=False)  # Covariance of Y past
        full_cov = np.cov(X_past, Y_past, rowvar=False)  # Full covariance matrix
        C_XYp[t] = full_cov[:model_order, model_order:]  # Cross-covariance X to Y
        C_YXp[t] = full_cov[model_order:, :model_order]  # Cross-covariance Y to X
        # C_XYp[t] = np.cov(X_p.T, Y_p.T)[:morder, morder:]
        # C_YXp[t] = np.cov(Y_p.T, X_p.T)[:morder, morder:]

        # Log warnings if NaN values are detected
        if np.any(np.isnan(cov_Xp[t])):
            logging.warning(f"NaN values detected in cov_Xp at time step {t}")

    return cov_Xp, cov_Yp, C_XYp, C_YXp

def estimate_coefficients(current_data_matrix: np.ndarray, lagged_data_matrix: np.ndarray, n_trials: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute regression coefficients and residual covariance for a VAR model.

    Parameters
    ----------
    current_data_matrix : np.ndarray
        Current time step data of shape (variables, trials).
    lagged_data_matrix : np.ndarray
        Lagged data of shape (variables * model_order, trials).
    n_trials : int
        Number of trials.

    Returns
    -------
    tuple
        A tuple containing:
        - coefficients : np.ndarray
            Regression coefficients (shape: (variables, variables * model_order + 1)).
        - residual_covariance : np.ndarray
            Residual covariance matrix (shape: (variables, variables)).

    Notes
    -----
    This function estimates coefficients using ordinary least squares (OLS) and includes a bias term.
    The lagged data is augmented with a column of ones to account for the intercept.
    """
    # Add bias term to lagged data
    lagged_data_with_bias = np.hstack([lagged_data_matrix, np.ones((n_trials, 1))])

    # Compute covariance matrices
    cross_cov_current = np.dot(current_data_matrix.T, current_data_matrix) / n_trials
    cross_cov_between = np.dot(current_data_matrix.T, lagged_data_with_bias) / n_trials
    auto_cov_lagged = np.dot(lagged_data_with_bias.T, lagged_data_with_bias) / n_trials

    # Regularize the auto-covariance matrix if singular
    auto_cov_lagged_reg = regularize_if_singular(auto_cov_lagged)
    if not np.allclose(auto_cov_lagged, auto_cov_lagged_reg):
        logging.warning("Applied regularization to auto_cov_lagged due to singularity")

    # Solve for coefficients using the regularized matrix
    coefficients = np.linalg.solve(auto_cov_lagged_reg, cross_cov_between.T).T

    # Compute residual covariance
    residual_covariance = cross_cov_current - np.dot(coefficients, np.dot(auto_cov_lagged_reg, coefficients.T))

    return coefficients, residual_covariance
