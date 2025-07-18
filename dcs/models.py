import logging
from typing import Dict, Tuple

import numpy as np

from .utils import compute_event_statistics
from .utils.preprocess import regularize_if_singular

logger = logging.getLogger(__name__)


def compute_multi_trial_BIC(event_data_max_order: np.ndarray, bic_params: Dict) -> Dict:
    """
    Calculate Bayesian Information Criterion (BIC) for multiple model orders across trial data.

    This function computes BIC values for Vector Autoregression (VAR) models of orders 1 to momax,
    supporting model selection with multiple BIC variants.

    Args:
        event_data_max_order (np.ndarray): Time series data for the maximum model order,
            shape (n_vars * (max_order + 1), n_observations, n_trials).
        bic_params (dict): Parameters for BIC calculation, containing:
            - "Params": Sub-dict with "BIC": {"momax": int}, the maximum model order.

    Returns:
        dict: Dictionary containing:
            - 'bic': BIC values for each model order and variant, shape (max_order, 4).
            - 'penalty_terms': Penalty terms for BIC, shape (max_order, 4).
            - 'log_likelihood': Log-likelihood for each model order, shape (max_order,).
            - 'sum_log_det_hessian': Sum of log determinant of Hessian, shape (max_order,).
            - 'optimal_orders': Optimal model orders for each BIC variant, shape (4,).

    Raises:
        ValueError: If input data shape is inconsistent with expected dimensions.
    """
    max_order = bic_params["Params"]["BIC"]["momax"]
    total_var_lag, n_observations, n_trials = event_data_max_order.shape
    n_vars = total_var_lag // (max_order + 1)

    if total_var_lag != n_vars * (max_order + 1):
        raise ValueError(
            "Data shape does not match expected dimensions based on max_order."
        )

    bic_outputs = {
        "bic": np.full((max_order, 4), np.nan),
        "pt_bic": np.full((max_order, 4), np.nan),
        "log_likelihood": np.full(max_order, np.nan),
        "sum_log_det_hessian": np.full(max_order, np.nan),
        "optimal_orders": None,
    }

    for model_order in range(1, max_order + 1):
        logger.info(f"Calculating BIC for model order: {model_order}")
        data_subset = event_data_max_order[: n_vars * (model_order + 1), :, :]

        log_likelihood, sum_log_det_hessian = compute_BIC_for_model(
            data_subset, model_order, bic_params
        )
        bic_outputs["log_likelihood"][model_order - 1] = log_likelihood
        bic_outputs["sum_log_det_hessian"][model_order - 1] = sum_log_det_hessian

        bic_outputs["pt_bic"][model_order - 1, 0] = (
            0.5 * n_observations * model_order * n_vars * n_vars * np.log(n_trials)
        )
        bic_outputs["pt_bic"][model_order - 1, 1] = 0.5 * sum_log_det_hessian
        bic_outputs["pt_bic"][model_order - 1, 2] = (
            0.5
            * n_observations
            * model_order
            * n_vars
            * n_vars
            * np.log(n_trials * n_observations)
        )
        bic_outputs["pt_bic"][model_order - 1, 3] = (
            0.5 * model_order * n_vars * n_vars * np.log(n_trials * n_observations)
        )

        for variant in range(4):
            bic_outputs["bic"][model_order - 1, variant] = (
                -bic_outputs["log_likelihood"][model_order - 1] * n_trials
                + bic_outputs["pt_bic"][model_order - 1, variant]
            )

    optimal_orders = np.nanargmin(bic_outputs["bic"], axis=0) + 1
    bic_outputs["mobic"] = optimal_orders
    logger.info(f"Optimal model orders for BIC variants: {optimal_orders}")

    return bic_outputs


def compute_BIC_for_model(
    event_data: np.ndarray, model_order: int, bic_params: Dict
) -> Tuple[float, float]:
    """
    Compute log-likelihood and sum of log determinant of Hessian for a specific model order.

    Supports 'biased' mode currently; 'debiased' mode is planned for future implementation.

    Args:
        event_data (np.ndarray): Event data for the model order,
            shape (n_vars * (model_order + 1), n_observations, n_trials).
        model_order (int): The VAR model order to evaluate.
        bic_params (dict): BIC parameters including:
            - 'Params': Sub-dict with 'BIC': {'mode': str}, e.g., 'biased'.
            - 'EstimMode': Estimation mode, either 'OLS' or 'RLS'.

    Returns:
        Tuple[float, float]:
            - log_likelihood: The log-likelihood of the model.
            - sum_log_det_hessian: Sum of the log determinant of the Hessian.

    Raises:
        ValueError: If 'mode' or 'EstimMode' is invalid.
    """
    total_var_lag, n_observations, n_trials = event_data.shape
    n_vars = total_var_lag // (model_order + 1)

    mode = bic_params["Params"]["BIC"]["mode"]
    if mode not in ["biased"]:
        raise ValueError(f"Unsupported mode '{mode}'; only 'biased' is implemented.")

    if mode == "biased":
        stats = compute_event_statistics(event_data, model_order)
    else:
        raise NotImplementedError("'debiased' mode is not yet supported.")

    log_det_hessian = np.zeros(n_observations)
    residual_determinants = np.zeros(n_observations)

    estim_mode = bic_params["EstimMode"]
    if estim_mode not in ["OLS", "RLS"]:
        raise ValueError(f"Invalid EstimMode '{estim_mode}'; must be 'OLS' or 'RLS'.")

    for t in range(n_observations):
        covariance_current = stats["Sigma"][t, n_vars:, n_vars:]
        residual_cov = stats[estim_mode]["Sigma_Et"][t]
        residual_determinants[t] = np.prod(np.diag(residual_cov))
        log_det_hessian[t] = (
            model_order * n_vars**2 * np.log(n_trials)
            + n_vars * np.log(np.linalg.det(covariance_current))
            - n_vars * model_order * np.log(residual_determinants[t] or 1e-8)
        )

    log_likelihood = (
        -0.5 * n_observations * n_vars * np.log(2 * np.pi)
        - 0.5 * np.sum(np.log(residual_determinants + 1e-8))
        - 0.5 * n_observations * n_vars
    )
    sum_log_det_hessian = np.sum(log_det_hessian)

    logger.debug(
        f"Model order {model_order}: log_likelihood={log_likelihood:.4f}, sum_log_det_hessian={sum_log_det_hessian:.4f}"
    )

    return log_likelihood, sum_log_det_hessian


def select_model_order(
    time_series_data: np.ndarray, max_model_order: int, time_mode: str
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Select the optimal VAR model order using Bayesian Information Criterion (BIC).

    Evaluates BIC scores for model orders 1 to max_model_order under specified time mode.

    Args:
        time_series_data (np.ndarray): Time series data, shape (n_vars, n_observations, n_trials).
        max_model_order (int): Maximum model order to evaluate.
        time_mode (str): 'inhomo' (inhomogeneous) or 'homo' (homogeneous).

    Returns:
        Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
            - bic_scores: BIC scores, shape (max_model_order, 2).
            - optimal_orders: Optimal model orders for each BIC variant, shape (2,).
            - log_likelihoods: Log-likelihoods, shape (max_model_order,).
            - penalty_terms: Penalty terms, shape (max_model_order, 2).

    Raises:
        ValueError: If time_mode is invalid or all BIC scores are NaN.
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
        )  # Adjusted based on data usage:‌ T = len(range(mo + 1, nobs))
        logger.info(f"Processing model order: {model_order}")

        _, _, log_likelihoods[model_order - 1], sum_log_det_hessian[model_order - 1] = (
            estimate_var_coefficients(
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


def estimate_var_coefficients(
    time_series_data: np.ndarray,
    model_order: int,
    max_model_order: int,
    time_mode: str,
    lag_mode: str,
    epsilon: float = 1e-8,
) -> Tuple[np.ndarray, np.ndarray, float, float]:
    """
    Estimate VAR coefficients and compute residual covariance with regularization.

    Supports both inhomogeneous ('inhomo') and homogeneous ('homo') time modes.

    Args:
        time_series_data (np.ndarray): Data array, shape (n_vars, n_observations, n_trials).
        model_order (int): Model order for the VAR process.
        max_model_order (int): Maximum model order for lag_mode 'infocrit'.
        time_mode (str): 'inhomo' or 'homo'.
        lag_mode (str): 'infocrit' or 'var' to determine lag structure.
        epsilon (float, optional): Regularization term for singular matrices. Defaults to 1e-8.

    Returns:
        Tuple[np.ndarray, np.ndarray, float, float]:
            - coefficients: VAR coefficients, shape varies by time_mode.
            - residual_covariance: Residual covariance, shape varies by time_mode.
            - log_likelihood: Log-likelihood of the model.
            - sum_log_det_hessian: Sum of log determinants of the Hessian.

    Raises:
        ValueError: If n_vars == 1, or time_mode/lag_mode is invalid.
    """
    # --- Start: Added Error Handling ---
    if not isinstance(time_series_data, np.ndarray):
        raise TypeError("Input 'time_series_data' must be a NumPy array.")

    if time_series_data.ndim != 3:
        raise ValueError(
            f"Input 'time_series_data' must be 3-dimensional (n_vars, n_observations, n_trials), got {time_series_data.ndim} dimensions."
        )

    n_vars, n_observations, n_trials = time_series_data.shape
    if n_vars <= 1:
        raise ValueError(
            f"Input must be multivariate (n_vars > 1), got n_vars={n_vars}."
        )

    if np.isnan(time_series_data).any() or np.isinf(time_series_data).any():
        # Optional: Check for NaN/Inf if they cause issues later
        logger.warning("Input 'time_series_data' contains NaN or Inf values.")

    if not isinstance(model_order, int) or model_order <= 0:
        raise ValueError("Input 'model_order' must be a positive integer.")

    if time_mode not in ["inhomo", "homo"]:
        raise ValueError("Invalid time_mode; must be 'inhomo' or 'homo'.")

    if lag_mode not in ["infocrit", "var"]:
        raise ValueError("Invalid lag_mode; must be 'infocrit' or 'var'.")

    if n_observations <= model_order:
        raise ValueError(
            f"Number of observations ({n_observations}) must be greater than the model order ({model_order})."
        )
    # --- End: Added Error Handling --

    lag_depth = max_model_order + 1 if lag_mode == "infocrit" else model_order + 1
    extended_data = np.zeros(
        (n_vars, lag_depth, n_observations + lag_depth - 1, n_trials)
    )
    for k in range(lag_depth):
        extended_data[:, k, k : k + n_observations, :] = time_series_data

    current_data = extended_data[:, 0, lag_depth - 1 : n_observations, :]
    lagged_data = extended_data[
        :, 1 : model_order + 1, lag_depth - 1 : n_observations, :
    ]
    n_time_steps = current_data.shape[1]

    coefficients = np.zeros((n_time_steps, n_vars, n_vars * model_order))
    residual_covariance = np.zeros((n_time_steps, n_vars, n_vars))
    residual_determinants = np.zeros(n_time_steps)
    log_det_hessian = np.zeros(n_time_steps)

    for t in range(n_time_steps):
        cov_current = np.dot(current_data[:, t, :], current_data[:, t, :].T) / n_trials
        lagged_matrix = np.vstack(
            [
                lagged_data[:, :model_order, t, :].reshape(
                    n_vars * model_order, n_trials
                ),
                np.ones((1, n_trials)),
            ]
        )
        cov_current_lagged = np.dot(current_data[:, t, :], lagged_matrix.T) / n_trials
        cov_lagged = np.dot(lagged_matrix, lagged_matrix.T) / n_trials
        cov_lagged_reg = regularize_if_singular(cov_lagged)
        coeff = np.linalg.solve(cov_lagged_reg, cov_current_lagged.T).T

        if time_mode == "inhomo":  # Should be aligned!
            coefficients[t, :, :] = coeff[:, :-1]
            residual_covariance[t, :, :] = cov_current - np.dot(
                coeff, np.dot(cov_lagged, coeff.T)
            )
            # residual_covariance[t, :, :] = cov_current - np.dot(coefficients[t, :, :].T, np.dot(cov_lagged, coefficients[t, :, :]))
            residual_determinants[t] = np.prod(np.diag(residual_covariance[t, :, :]))
            lagged_cov_subset = cov_lagged[:-1, :-1] * n_trials
            determinant_hessian = np.linalg.det(
                lagged_cov_subset * n_trials
            ) ** n_vars * (1 / residual_determinants[t]) ** (n_vars * model_order)
            log_det_hessian[t] = (
                model_order * n_vars**2 * np.log(n_trials)
                + n_vars * np.log(np.linalg.det(lagged_cov_subset))
                - n_vars * model_order * np.log(residual_determinants[t] or epsilon)
            )

    if time_mode == "inhomo":
        determinants_clamped = np.where(
            residual_determinants < epsilon, epsilon, residual_determinants
        )
        log_likelihood = (
            -0.5 * n_time_steps * n_vars * np.log(2 * np.pi)
            - 0.5 * np.sum(np.log(determinants_clamped))
            - 0.5 * n_time_steps * n_vars
        )
        sum_log_det_hessian = np.sum(log_det_hessian)

    elif time_mode == "homo":
        cov_current_mean = np.mean(cov_current, axis=0)
        cov_current_lagged_mean = np.mean(cov_current_lagged, axis=0)
        cov_lagged_mean = np.mean(cov_lagged, axis=0)
        # coefficients = np.linalg.solve(cov_lagged_mean, cov_current_lagged_mean.T).T[:, :-1]
        coefficients = np.dot(
            cov_current_lagged_mean.reshape(n_vars, n_vars * model_order),
            np.linalg.inv(cov_lagged_mean),
        )
        residual_covariance = cov_current_mean - np.dot(
            coefficients, np.dot(cov_lagged_mean, coefficients.T)
        )
        determinant = np.prod(np.diag(residual_covariance))
        determinant_hessian = np.linalg.det(
            cov_lagged_mean * n_time_steps * n_trials
        ) ** n_vars * (1 / determinant) ** (n_vars * model_order)
        log_likelihood = (
            -0.5 * n_time_steps * n_vars * np.log(2 * np.pi)
            - 0.5 * n_time_steps * np.log(determinant or epsilon)
            - 0.5 * n_time_steps * n_vars
        )
        sum_log_det_hessian = n_vars * np.log(
            np.linalg.det(cov_lagged_mean * n_time_steps * n_trials)
        ) + n_vars * model_order * np.log(1 / (determinant or epsilon))

    return coefficients, residual_covariance, log_likelihood, sum_log_det_hessian
