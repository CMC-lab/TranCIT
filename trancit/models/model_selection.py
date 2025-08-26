"""
Model selection utilities for Dynamic Causal Strength (DCS).

This module provides high-level model selection utilities including
BIC-based model order selection and related functionality.
"""

import logging
from typing import Tuple

import numpy as np

from .var_estimation import VAREstimator

logger = logging.getLogger(__name__)


def select_model_order(
    time_series_data: np.ndarray, max_model_order: int, time_mode: str
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Select the optimal VAR model order using Bayesian Information Criterion (BIC).

    Evaluates BIC scores for model orders 1 to max_model_order under specified time mode.

    Parameters
    ----------
    time_series_data : np.ndarray
        Time series data, shape (n_vars, n_observations, n_trials).
    max_model_order : int
        Maximum model order to evaluate.
    time_mode : str
        'inhomo' (inhomogeneous) or 'homo' (homogeneous).

    Returns
    -------
    Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]
        - bic_scores: BIC scores, shape (max_model_order, 2).
        - optimal_orders: Optimal model orders for each BIC variant, shape (2,).
        - log_likelihoods: Log-likelihoods, shape (max_model_order,).
        - penalty_terms: Penalty terms, shape (max_model_order, 2).

    Raises
    ------
    ValueError
        If time_mode is invalid or all BIC scores are NaN.

    Examples
    --------
    >>> import numpy as np
    >>> data = np.random.randn(2, 1000, 20)  # (n_vars, n_obs, n_trials)
    >>> bic_scores, optimal_orders, log_likelihoods, penalty_terms = select_model_order(
    ...     data, max_model_order=6, time_mode="inhomo"
    ... )
    >>> print(f"Optimal model orders: {optimal_orders}")
    """
    if time_mode not in ["inhomo", "homo"]:
        raise ValueError("time_mode must be 'inhomo' or 'homo'.")

    n_vars, n_observations, n_trials = time_series_data.shape
    bic_scores = np.full((max_model_order, 2), np.nan)
    penalty_terms = np.full((max_model_order, 2), np.nan)
    log_likelihoods = np.full(max_model_order, np.nan)
    sum_log_det_hessian = np.full(max_model_order, np.nan)

    for model_order in range(1, max_model_order + 1):
        n_time_steps = (
            n_observations - model_order - 1
        )  # Adjusted based on data usage: T = len(range(mo + 1, nobs))
        logger.info(f"Processing model order: {model_order}")

        # Create VAREstimator instance for this model order
        estimator = VAREstimator(model_order=model_order, time_mode=time_mode)

        _, _, log_likelihoods[model_order - 1], sum_log_det_hessian[model_order - 1] = (
            estimator.estimate_var_coefficients(
                time_series_data, model_order, max_model_order, time_mode, "infocrit"
            )
        )

        penalty_terms[model_order - 1, 1] = sum_log_det_hessian[model_order - 1]

        if time_mode == "inhomo":
            penalty_terms[model_order - 1, 0] = (
                n_time_steps * model_order * n_vars * n_vars * np.log(n_trials)
            )
            bic_scores[model_order - 1, 0] = (
                -log_likelihoods[model_order - 1] * n_trials
                + penalty_terms[model_order - 1, 0]
            )
            bic_scores[model_order - 1, 1] = (
                -log_likelihoods[model_order - 1] * n_trials
                + sum_log_det_hessian[model_order - 1]
            )
        elif time_mode == "homo":
            penalty_terms[model_order - 1, 0] = (
                model_order * n_vars * n_vars * np.log(n_time_steps * n_trials)
            )
            bic_scores[model_order - 1, 0] = (
                -log_likelihoods[model_order - 1] * n_trials
                + penalty_terms[model_order - 1, 0]
            )
            bic_scores[model_order - 1, 1] = (
                -log_likelihoods[model_order - 1] * n_trials
                + sum_log_det_hessian[model_order - 1]
            )

    if np.isnan(bic_scores).all():
        raise ValueError(
            "All BIC scores are NaN; verify input data or reduce max_model_order."
        )

    optimal_orders = np.nanargmin(bic_scores, axis=0) + 1
    logger.info(f"Optimal model orders: {optimal_orders}")

    return bic_scores, optimal_orders, log_likelihoods, penalty_terms
