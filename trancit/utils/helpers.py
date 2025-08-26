"""
Helper functions for Dynamic Causal Strength (DCS).

This module provides utility functions for covariance computation, coefficient estimation,
and multi-variable linear regression. These functions support the core DCS algorithms.
"""

import logging
from typing import Tuple

import numpy as np

from ..core.exceptions import ComputationError, ValidationError
from .preprocess import regularize_if_singular

logger = logging.getLogger(__name__)


def compute_covariances(
    lagged_data_array: np.ndarray, n_time_steps: int, model_order: int
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute covariance matrices for lagged data.

    This function computes covariance matrices for each time step using NumPy's `np.cov`.
    It assumes two variables (Y and X) and processes their lagged data accordingly.

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
    Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]
        A tuple containing:
        - cov_Xp : np.ndarray
            Covariance of X past (shape: (time_steps, model_order, model_order)).
        - cov_Yp : np.ndarray
            Covariance of Y past (shape: (time_steps, model_order, model_order)).
        - C_XYp : np.ndarray
            Cross-covariance X past to Y past (shape: (time_steps, model_order, model_order)).
        - C_YXp : np.ndarray
            Cross-covariance Y past to X past (shape: (time_steps, model_order, model_order)).

    Raises
    ------
    ValidationError
        If input parameters are invalid.
    ComputationError
        If computation fails due to numerical issues.

    Examples
    --------
    >>> import numpy as np
    >>> lagged_data = np.random.randn(2, 4, 100, 10)  # (vars, morder, time, trials)
    >>> cov_Xp, cov_Yp, C_XYp, C_YXp = compute_covariances(lagged_data, 100, 4)
    >>> print(f"Covariance shapes: {cov_Xp.shape}, {cov_Yp.shape}")
    """
    if not isinstance(lagged_data_array, np.ndarray) or lagged_data_array.ndim != 4:
        raise ValidationError(
            "lagged_data_array must be a 4D numpy array",
            "lagged_data_ndim",
            lagged_data_array.ndim if hasattr(lagged_data_array, "ndim") else None,
        )

    if not isinstance(n_time_steps, int) or n_time_steps <= 0:
        raise ValidationError(
            "n_time_steps must be a positive integer", "n_time_steps", n_time_steps
        )

    if not isinstance(model_order, int) or model_order <= 0:
        raise ValidationError(
            "model_order must be a positive integer", "model_order", model_order
        )

    if lagged_data_array.shape[0] != 2:
        raise ValidationError(
            "lagged_data_array must have exactly 2 variables",
            "n_vars",
            lagged_data_array.shape[0],
        )

    if lagged_data_array.shape[1] != model_order:
        raise ValidationError(
            "lagged_data_array model_order dimension doesn't match",
            "model_order_dim",
            lagged_data_array.shape[1],
        )

    if lagged_data_array.shape[2] != n_time_steps:
        raise ValidationError(
            "lagged_data_array time dimension doesn't match n_time_steps",
            "time_dim",
            lagged_data_array.shape[2],
        )

    if lagged_data_array.shape[3] < 2:
        raise ValidationError(
            "At least 2 trials required for covariance computation",
            "n_trials",
            lagged_data_array.shape[3],
        )

    try:
        cov_Xp = np.zeros((n_time_steps, model_order, model_order))
        cov_Yp = np.zeros((n_time_steps, model_order, model_order))
        C_XYp = np.zeros((n_time_steps, model_order, model_order))
        C_YXp = np.zeros((n_time_steps, model_order, model_order))

        for t in range(n_time_steps):
            try:
                X_past = lagged_data_array[1, :, t, :].T
                Y_past = lagged_data_array[0, :, t, :].T

                cov_Xp[t] = np.cov(X_past, rowvar=False)  # Covariance of X past
                cov_Yp[t] = np.cov(Y_past, rowvar=False)  # Covariance of Y past
                full_cov = np.cov(X_past, Y_past, rowvar=False)
                C_XYp[t] = full_cov[
                    :model_order, model_order:
                ]  # Cross-covariance X to Y
                C_YXp[t] = full_cov[
                    model_order:, :model_order
                ]  # Cross-covariance Y to X

                if np.any(np.isnan(cov_Xp[t])):
                    logger.error(f"NaN values detected in cov_Xp at time step {t}")

            except Exception as e:
                logger.error(f"Failed to compute covariances at time step {t}: {e}")
                cov_Xp[t] = np.nan
                cov_Yp[t] = np.nan
                C_XYp[t] = np.nan
                C_YXp[t] = np.nan

        logger.info(f"Successfully computed covariances for {n_time_steps} time steps")
        return cov_Xp, cov_Yp, C_XYp, C_YXp

    except Exception as e:
        logger.error(f"Covariance computation failed: {e}")
        raise ComputationError(
            f"Covariance computation failed: {e}", "covariance_computation", ()
        )


def estimate_coefficients(
    current_data_matrix: np.ndarray, lagged_data_matrix: np.ndarray, n_trials: int
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute regression coefficients and residual covariance for a VAR model.

    This function estimates coefficients using ordinary least squares (OLS)
    and includes a bias term.
    The lagged data is augmented with a column of ones to account for the intercept.

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
    Tuple[np.ndarray, np.ndarray]
        A tuple containing:
        - coefficients : np.ndarray
            Regression coefficients (shape: (variables, variables * model_order + 1)).
        - residual_covariance : np.ndarray
            Residual covariance matrix (shape: (variables, variables)).

    Raises
    ------
    ValidationError
        If input parameters are invalid.
    ComputationError
        If computation fails due to numerical issues.

    Examples
    --------
    >>> import numpy as np
    >>> current_data = np.random.randn(2, 10)  # (vars, trials)
    >>> lagged_data = np.random.randn(8, 10)   # (vars * morder, trials)
    >>> coeff, resid_cov = estimate_coefficients(current_data, lagged_data, 10)
    >>> print(f"Coefficients shape: {coeff.shape}")
    >>> print(f"Residual covariance shape: {resid_cov.shape}")
    """

    n_vars = current_data_matrix.shape[0]

    try:
        lagged_data_augmented = np.hstack([lagged_data_matrix, np.ones((n_trials, 1))])

        cross_cov_current = (
            np.dot(current_data_matrix.T, current_data_matrix) / n_trials
        )
        cross_cov_between = (
            np.dot(current_data_matrix.T, lagged_data_augmented) / n_trials
        )
        auto_cov_lagged = (
            np.dot(lagged_data_augmented.T, lagged_data_augmented) / n_trials
        )

        auto_cov_lagged_reg = regularize_if_singular(auto_cov_lagged)
        if not np.allclose(auto_cov_lagged, auto_cov_lagged_reg):
            logger.warning(
                "Applied regularization to auto_cov_lagged due to singularity"
            )

        coefficients = np.linalg.solve(auto_cov_lagged_reg, cross_cov_between.T).T

        residual_covariance = cross_cov_current - np.dot(
            coefficients, np.dot(auto_cov_lagged_reg, coefficients.T)
        )
        logger.info(f"Successfully estimated coefficients for {n_vars} variables")
        return coefficients, residual_covariance

    except Exception as e:
        logger.error(f"Coefficient estimation failed: {e}")
        raise ComputationError(
            f"Coefficient estimation failed: {e}", "coefficient_estimation", None
        )


def compute_multi_variable_linear_regression(
    X: np.ndarray, Y: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Perform multiple linear regressions for each slice of Y against X.

    This function efficiently fits a linear model Y_slice = intercept + coeff * X
    for each slice Y[:, i, j] using a vectorized approach. The implementation uses
    numpy's lstsq solver for numerical stability.

    Parameters
    ----------
    X : np.ndarray
        Independent variable, 1D array of shape (N,).
    Y : np.ndarray
        Dependent variable, 3D array of shape (N, N2, N3).
        N must match the length of X. Each Y[:, i, j] slice is regressed against X.

    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        A tuple containing:
        - coeff (np.ndarray): Regression slopes (coefficients for X),
                             shape (N2, N3).
        - intercept (np.ndarray): Regression intercepts, shape (N2, N3).

    Raises
    ------
    ValueError
        If X and Y do not have the same first dimension length,
        if N < 2 (at least two data points required),
        if X or Y contain NaN values, or
        if the regression problem is ill-conditioned.

    Notes
    -----
    The function uses a vectorized approach for efficiency, solving all regressions
    simultaneously. It includes checks for numerical stability and proper conditioning
    of the regression problem.

    Examples
    --------
    >>> X = np.linspace(0, 10, 100)
    >>> Y = np.random.randn(100, 5, 3)  # 100 points, 5x3 slices
    >>> coeff, intercept = compute_multi_variable_linear_regression(X, Y)
    >>> print(f"Coefficients shape: {coeff.shape}")  # (5, 3)
    >>> print(f"Intercepts shape: {intercept.shape}")  # (5, 3)
    """
    if Y.ndim != 3:
        raise ValueError(f"Dependent variable Y must be 3-dimensional, got {Y.ndim}.")

    N, N2, N3 = Y.shape

    if X.ndim != 1 or X.shape[0] != N:
        raise ValueError(
            f"Independent variable X must be 1D with length {N} "
            f"to match Y's first dimension, got shape {X.shape}."
        )
    if N < 2:
        raise ValueError(
            "At least two data points (N >= 2) are required for linear regression."
        )
    if np.any(np.isnan(X)) or np.any(np.isnan(Y)):
        raise ValueError("Input arrays X and Y must not contain NaN values.")

    X_design = np.vstack([np.ones(N), X]).T
    Y_flat = Y.reshape(N, -1)

    betas, _, rank, _ = np.linalg.lstsq(X_design, Y_flat, rcond=None)

    if rank < 2:
        raise ValueError(
            "Regression problem is ill-conditioned. Check your input data."
        )

    intercept = betas[0].reshape(N2, N3)
    coeff = betas[1].reshape(N2, N3)

    return coeff, intercept
