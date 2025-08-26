"""
VAR (Vector Autoregressive) model estimation.

This module provides robust VAR model estimation with proper
error handling and validation.
"""

import logging
from typing import Tuple

import numpy as np

from ..core.exceptions import DataError, ValidationError
from ..utils.preprocess import regularize_if_singular

logger = logging.getLogger(__name__)


class VAREstimator:
    """
    Vector Autoregressive (VAR) model estimator.

    This class provides methods for estimating VAR model parameters
    including coefficients, residuals, and covariance matrices.
    """

    def __init__(self, model_order: int, time_mode: str = "inhomo"):
        """
        Initialize VAR estimator.

        Parameters
        ----------
        model_order : int
            Model order for VAR analysis
        time_mode : str
            Time mode: 'inhomo' (time-inhomogeneous) or 'homo' (homogeneous)
        """
        self.model_order = model_order
        self.time_mode = time_mode
        self._validate_parameters()

    def _validate_parameters(self) -> None:
        """Validate initialization parameters."""
        if self.model_order <= 0:
            raise ValidationError(
                "model_order must be positive", "model_order", self.model_order
            )
        if self.time_mode not in ["inhomo", "homo"]:
            raise ValidationError(
                "time_mode must be 'inhomo' or 'homo'", "time_mode", self.time_mode
            )

    def estimate_var_coefficients(
        self,
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
        self._validate_input_data(time_series_data, model_order, time_mode, lag_mode)

        n_vars, n_observations, n_trials = time_series_data.shape

        extended_data = self._prepare_extended_data(
            time_series_data, model_order, max_model_order, lag_mode
        )

        current_data, lagged_data, n_time_steps = self._extract_data_components(
            extended_data, model_order, max_model_order, lag_mode, n_observations
        )

        coefficients, residual_covariance, residual_determinants, log_det_hessian = (
            self._initialize_output_arrays(n_time_steps, n_vars, model_order)
        )

        for t in range(n_time_steps):
            coeff, cov, det, hessian = self._estimate_single_time_step(
                current_data, lagged_data, t, n_vars, model_order, n_trials, epsilon
            )
            coefficients[t] = coeff
            residual_covariance[t] = cov
            residual_determinants[t] = det
            log_det_hessian[t] = hessian

        if time_mode == "inhomo":
            log_likelihood, sum_log_det_hessian = (
                self._compute_inhomogeneous_statistics(
                    residual_determinants, n_time_steps, n_vars, epsilon
                )
            )
        else:  # time_mode == "homo"
            coefficients, residual_covariance, log_likelihood, sum_log_det_hessian = (
                self._compute_homogeneous_statistics(
                    current_data,
                    lagged_data,
                    n_vars,
                    model_order,
                    n_time_steps,
                    n_trials,
                    epsilon,
                )
            )

        return coefficients, residual_covariance, log_likelihood, sum_log_det_hessian

    def _validate_input_data(
        self,
        time_series_data: np.ndarray,
        model_order: int,
        time_mode: str,
        lag_mode: str,
    ) -> None:
        """Validate input data and parameters."""
        if not isinstance(time_series_data, np.ndarray):
            raise ValidationError(
                "time_series_data must be a NumPy array",
                "time_series_data_type",
                type(time_series_data),
            )

        if time_series_data.ndim != 3:
            raise ValidationError(
                "time_series_data must be 3-dimensional (n_vars, n_observations, n_trials)",
                "time_series_data_ndim",
                time_series_data.ndim,
            )

        n_vars, n_observations, n_trials = time_series_data.shape
        if n_vars <= 1:
            raise ValidationError(
                "Input must be multivariate (n_vars > 1)", "n_vars", n_vars
            )

        if np.isnan(time_series_data).any() or np.isinf(time_series_data).any():
            raise DataError(
                "time_series_data contains NaN or Inf values",
                time_series_data.shape,
                "float64",
            )

        if not isinstance(model_order, int) or model_order <= 0:
            raise ValidationError(
                "model_order must be a positive integer", "model_order", model_order
            )

        if time_mode not in ["inhomo", "homo"]:
            raise ValidationError(
                "time_mode must be 'inhomo' or 'homo'", "time_mode", time_mode
            )

        if lag_mode not in ["infocrit", "var"]:
            raise ValidationError(
                "lag_mode must be 'infocrit' or 'var'", "lag_mode", lag_mode
            )

        if n_observations <= model_order:
            raise ValidationError(
                f"Number of observations ({n_observations}) must be greater "
                f"than model_order ({model_order})",
                "n_observations_vs_model_order",
                (n_observations, model_order),
            )

    def _prepare_extended_data(
        self,
        time_series_data: np.ndarray,
        model_order: int,
        max_model_order: int,
        lag_mode: str,
    ) -> np.ndarray:
        """Prepare extended data structure for VAR estimation."""
        n_vars, n_observations, n_trials = time_series_data.shape
        lag_depth = max_model_order + 1 if lag_mode == "infocrit" else model_order + 1

        extended_data = np.zeros(
            (n_vars, lag_depth, n_observations + lag_depth - 1, n_trials)
        )

        for k in range(lag_depth):
            extended_data[:, k, k : k + n_observations, :] = time_series_data

        return extended_data

    def _extract_data_components(
        self,
        extended_data: np.ndarray,
        model_order: int,
        max_model_order: int,
        lag_mode: str,
        n_observations: int,
    ) -> Tuple[np.ndarray, np.ndarray, int]:
        """Extract current and lagged data components."""
        lag_depth = max_model_order + 1 if lag_mode == "infocrit" else model_order + 1

        current_data = extended_data[:, 0, lag_depth - 1 : n_observations, :]
        lagged_data = extended_data[
            :, 1 : model_order + 1, lag_depth - 1 : n_observations, :
        ]
        n_time_steps = current_data.shape[1]

        return current_data, lagged_data, n_time_steps

    def _initialize_output_arrays(
        self, n_time_steps: int, n_vars: int, model_order: int
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Initialize output arrays for VAR estimation."""
        coefficients = np.zeros((n_time_steps, n_vars, n_vars * model_order))
        residual_covariance = np.zeros((n_time_steps, n_vars, n_vars))
        residual_determinants = np.zeros(n_time_steps)
        log_det_hessian = np.zeros(n_time_steps)

        return coefficients, residual_covariance, residual_determinants, log_det_hessian

    def _estimate_single_time_step(
        self,
        current_data: np.ndarray,
        lagged_data: np.ndarray,
        t: int,
        n_vars: int,
        model_order: int,
        n_trials: int,
        epsilon: float,
    ) -> Tuple[np.ndarray, np.ndarray, float, float]:
        """Estimate VAR coefficients for a single time step."""
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

        residual_cov = cov_current - np.dot(coeff, np.dot(cov_lagged, coeff.T))

        residual_det = np.prod(np.diag(residual_cov))
        lagged_cov_subset = cov_lagged[:-1, :-1] * n_trials

        log_det_hessian = (
            model_order * n_vars**2 * np.log(n_trials)
            + n_vars * np.log(np.linalg.det(lagged_cov_subset))
            - n_vars * model_order * np.log(residual_det or epsilon)
        )

        return coeff[:, :-1], residual_cov, residual_det, log_det_hessian

    def _compute_inhomogeneous_statistics(
        self,
        residual_determinants: np.ndarray,
        n_time_steps: int,
        n_vars: int,
        epsilon: float,
    ) -> Tuple[float, float]:
        """Compute statistics for inhomogeneous time mode."""
        determinants_clamped = np.where(
            residual_determinants < epsilon, epsilon, residual_determinants
        )

        log_likelihood = (
            -0.5 * n_time_steps * n_vars * np.log(2 * np.pi)
            - 0.5 * np.sum(np.log(determinants_clamped))
            - 0.5 * n_time_steps * n_vars
        )

        return log_likelihood, np.sum(residual_determinants)

    def _compute_homogeneous_statistics(
        self,
        current_data: np.ndarray,
        lagged_data: np.ndarray,
        n_vars: int,
        model_order: int,
        n_time_steps: int,
        n_trials: int,
        epsilon: float,
    ) -> Tuple[np.ndarray, np.ndarray, float, float]:
        """Compute statistics for homogeneous time mode."""
        cov_current_mean = np.mean(current_data, axis=0)
        cov_current_lagged_mean = np.mean(lagged_data, axis=0)
        cov_lagged_mean = np.mean(lagged_data, axis=0)

        coefficients = np.dot(
            cov_current_lagged_mean.reshape(n_vars, n_vars * model_order),
            np.linalg.inv(cov_lagged_mean),
        )

        residual_covariance = cov_current_mean - np.dot(
            coefficients, np.dot(cov_lagged_mean, coefficients.T)
        )

        determinant = np.prod(np.diag(residual_covariance))

        log_likelihood = (
            -0.5 * n_time_steps * n_vars * np.log(2 * np.pi)
            - 0.5 * n_time_steps * np.log(determinant or epsilon)
            - 0.5 * n_time_steps * n_vars
        )

        sum_log_det_hessian = n_vars * np.log(
            np.linalg.det(cov_lagged_mean * n_time_steps * n_trials)
        ) + n_vars * model_order * np.log(1 / (determinant or epsilon))

        return coefficients, residual_covariance, log_likelihood, sum_log_det_hessian
