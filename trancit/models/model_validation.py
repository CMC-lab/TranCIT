"""
Model validation utilities for Dynamic Causal Strength (DCS).

This module provides tools for validating VAR models and assessing
their quality and reliability.
"""

import logging
from typing import Dict

import numpy as np
from scipy import stats

from trancit.core.base import BaseAnalyzer, BaseResult
from trancit.core.exceptions import ComputationError, ValidationError

logger = logging.getLogger(__name__)


class ModelValidationResult(BaseResult):
    """Result container for model validation analysis."""

    def __init__(
        self,
        residual_autocorrelation: np.ndarray,
        residual_normality: Dict[str, float],
        model_stability: bool,
        condition_number: float,
        log_likelihood: float,
        aic_score: float,
        bic_score: float,
    ):
        """
        Initialize model validation result.

        Parameters
        ----------
        residual_autocorrelation : np.ndarray
            Autocorrelation of residuals
        residual_normality : Dict[str, float]
            Normality test results (p-values)
        model_stability : bool
            Whether the model is stable
        condition_number : float
            Condition number of the design matrix
        log_likelihood : float
            Log-likelihood of the model
        aic_score : float
            Akaike Information Criterion score
        bic_score : float
            Bayesian Information Criterion score
        """
        super().__init__(
            residual_autocorrelation=residual_autocorrelation,
            residual_normality=residual_normality,
            model_stability=model_stability,
            condition_number=condition_number,
            log_likelihood=log_likelihood,
            aic_score=aic_score,
            bic_score=bic_score,
        )


class ModelValidator(BaseAnalyzer):
    """
    Model validator for VAR models.

    This class provides comprehensive validation tools for assessing
    the quality and reliability of fitted VAR models.
    """

    def __init__(
        self,
        max_lag: int = 10,
        significance_level: float = 0.05,
        stability_threshold: float = 1.0,
        **kwargs,
    ):
        """
        Initialize model validator.

        Parameters
        ----------
        max_lag : int
            Maximum lag for autocorrelation testing
        significance_level : float
            Significance level for hypothesis tests
        stability_threshold : float
            Threshold for model stability (eigenvalues)
        **kwargs
            Additional configuration parameters
        """
        super().__init__(
            max_lag=max_lag,
            significance_level=significance_level,
            stability_threshold=stability_threshold,
            **kwargs,
        )

    def validate(
        self,
        coefficients: np.ndarray,
        residuals: np.ndarray,
        design_matrix: np.ndarray,
        n_observations: int,
        n_parameters: int,
        **kwargs,
    ) -> ModelValidationResult:
        """
        Perform comprehensive model validation.

        Parameters
        ----------
        coefficients : np.ndarray
            VAR coefficients
        residuals : np.ndarray
            Model residuals
        design_matrix : np.ndarray
            Design matrix used for estimation
        n_observations : int
            Number of observations
        n_parameters : int
            Number of parameters in the model
        **kwargs
            Additional parameters

        Returns
        -------
        ModelValidationResult
            Model validation results
        """
        self._validate_input_data(coefficients, residuals, design_matrix)
        self._log_analysis_start(coefficients.shape)

        try:
            residual_autocorr = self._check_residual_autocorrelation(residuals)

            residual_normality = self._check_residual_normality(residuals)

            model_stability = self._check_model_stability(coefficients)

            condition_number = self._compute_condition_number(design_matrix)

            log_likelihood = self._compute_log_likelihood(residuals)
            aic_score = self._compute_aic(log_likelihood, n_parameters, n_observations)
            bic_score = self._compute_bic(log_likelihood, n_parameters, n_observations)

            self._log_analysis_complete()

            return ModelValidationResult(
                residual_autocorrelation=residual_autocorr,
                residual_normality=residual_normality,
                model_stability=model_stability,
                condition_number=condition_number,
                log_likelihood=log_likelihood,
                aic_score=aic_score,
                bic_score=bic_score,
            )

        except Exception as e:
            logger.error(f"Model validation failed: {e}")
            raise ComputationError(
                f"Model validation failed: {e}",
                "validation_computation",
                coefficients.shape,
            )

    def _validate_config(self) -> None:
        """Validate configuration parameters."""
        if self.config["max_lag"] <= 0:
            raise ValidationError(
                "max_lag must be positive", "max_lag", self.config["max_lag"]
            )

        if not 0 < self.config["significance_level"] < 1:
            raise ValidationError(
                "significance_level must be between 0 and 1",
                "significance_level",
                self.config["significance_level"],
            )

        if self.config["stability_threshold"] <= 0:
            raise ValidationError(
                "stability_threshold must be positive",
                "stability_threshold",
                self.config["stability_threshold"],
            )

    def _validate_input_data(  # type: ignore[override]
        self, coefficients: np.ndarray, residuals: np.ndarray, design_matrix: np.ndarray
    ) -> None:
        """Validate input data format and dimensions."""
        super()._validate_input_data(coefficients)

        if not isinstance(residuals, np.ndarray):
            raise ValidationError(
                "residuals must be a NumPy array", "residuals_type", type(residuals)
            )

        if not isinstance(design_matrix, np.ndarray):
            raise ValidationError(
                "design_matrix must be a NumPy array",
                "design_matrix_type",
                type(design_matrix),
            )

        if np.any(np.isnan(coefficients)) or np.any(np.isinf(coefficients)):
            raise ValidationError("coefficients contains NaN or Inf values")

        if np.any(np.isnan(residuals)) or np.any(np.isinf(residuals)):
            raise ValidationError("residuals contains NaN or Inf values")

    def _check_residual_autocorrelation(self, residuals: np.ndarray) -> np.ndarray:
        """
        Check for residual autocorrelation.

        Parameters
        ----------
        residuals : np.ndarray
            Model residuals

        Returns
        -------
        np.ndarray
            Autocorrelation coefficients for different lags
        """
        max_lag = min(self.config["max_lag"], residuals.shape[0] // 4)
        autocorr = np.zeros(max_lag)

        for lag in range(1, max_lag + 1):
            var_autocorr = []
            for var in range(residuals.shape[1]):
                series = residuals[:, var]
                if len(series) > lag:
                    corr = np.corrcoef(series[:-lag], series[lag:])[0, 1]
                    if not np.isnan(corr):
                        var_autocorr.append(corr)

            if var_autocorr:
                autocorr[lag - 1] = np.mean(var_autocorr)

        return autocorr

    def _check_residual_normality(self, residuals: np.ndarray) -> Dict[str, float]:
        """
        Check residual normality using various tests.

        Parameters
        ----------
        residuals : np.ndarray
            Model residuals

        Returns
        -------
        Dict[str, float]
            Dictionary with test names and p-values
        """
        normality_tests = {}

        for var in range(residuals.shape[1]):
            series = residuals[:, var]

            series = series[~np.isnan(series)]

            if len(series) > 3:
                try:
                    _, p_value = stats.shapiro(series)
                    normality_tests[f"shapiro_var_{var}"] = p_value
                except (ValueError, RuntimeWarning):
                    normality_tests[f"shapiro_var_{var}"] = np.nan

                try:
                    result = stats.anderson(series)
                    normality_tests[f"anderson_var_{var}"] = result.significance_level[
                        2
                    ]  # 5% level
                except (ValueError, IndexError):
                    normality_tests[f"anderson_var_{var}"] = np.nan

        return normality_tests

    def _check_model_stability(self, coefficients: np.ndarray) -> bool:
        """
        Check VAR model stability by examining eigenvalues.

        Parameters
        ----------
        coefficients : np.ndarray
            VAR coefficients

        Returns
        -------
        bool
            True if model is stable, False otherwise
        """
        try:
            n_vars = coefficients.shape[0]
            model_order = (coefficients.shape[1] - 1) // n_vars

            companion_size = n_vars * model_order
            companion_matrix = np.zeros((companion_size, companion_size))

            for i in range(model_order - 1):
                companion_matrix[
                    i * n_vars : (i + 1) * n_vars, (i + 1) * n_vars : (i + 2) * n_vars
                ] = np.eye(n_vars)

            for i in range(model_order):
                start_idx = i * n_vars
                end_idx = (i + 1) * n_vars
                companion_matrix[(model_order - 1) * n_vars :, start_idx:end_idx] = (
                    coefficients[:, i * n_vars : (i + 1) * n_vars]
                )

            eigenvalues = np.linalg.eigvals(companion_matrix)
            max_eigenvalue = np.max(np.abs(eigenvalues))

            return max_eigenvalue < self.config["stability_threshold"]

        except Exception as e:
            logger.warning(f"Model stability check failed: {e}")
            return False

    def _compute_condition_number(self, design_matrix: np.ndarray) -> float:
        """
        Compute condition number of the design matrix.

        Parameters
        ----------
        design_matrix : np.ndarray
            Design matrix

        Returns
        -------
        float
            Condition number
        """
        try:
            singular_values = np.linalg.svd(design_matrix, compute_uv=False)
            condition_number = singular_values[0] / singular_values[-1]
            return condition_number
        except Exception as e:
            logger.warning(f"Condition number computation failed: {e}")
            return np.inf

    def _compute_log_likelihood(self, residuals: np.ndarray) -> float:
        """
        Compute log-likelihood of the model.

        Parameters
        ----------
        residuals : np.ndarray
            Model residuals

        Returns
        -------
        float
            Log-likelihood value
        """
        try:
            residual_cov = np.cov(residuals.T)

            n_obs = residuals.shape[0]
            n_vars = residuals.shape[1]

            log_det = np.log(np.linalg.det(residual_cov))

            inv_cov = np.linalg.inv(residual_cov)
            mahal_dist = np.sum(residuals @ inv_cov * residuals, axis=1)

            log_likelihood = (
                -0.5
                * n_obs
                * (n_vars * np.log(2 * np.pi) + log_det + np.mean(mahal_dist))
            )

            return log_likelihood

        except Exception as e:
            logger.warning(f"Log-likelihood computation failed: {e}")
            return np.nan

    def _compute_aic(
        self, log_likelihood: float, n_parameters: int, n_observations: int
    ) -> float:
        """
        Compute Akaike Information Criterion.

        Parameters
        ----------
        log_likelihood : float
            Log-likelihood of the model
        n_parameters : int
            Number of parameters
        n_observations : int
            Number of observations

        Returns
        -------
        float
            AIC score
        """
        if np.isnan(log_likelihood):
            return np.nan

        return 2 * n_parameters - 2 * log_likelihood

    def _compute_bic(
        self, log_likelihood: float, n_parameters: int, n_observations: int
    ) -> float:
        """
        Compute Bayesian Information Criterion.

        Parameters
        ----------
        log_likelihood : float
            Log-likelihood of the model
        n_parameters : int
            Number of parameters
        n_observations : int
            Number of observations

        Returns
        -------
        float
            BIC score
        """
        if np.isnan(log_likelihood):
            return np.nan

        return np.log(n_observations) * n_parameters - 2 * log_likelihood
