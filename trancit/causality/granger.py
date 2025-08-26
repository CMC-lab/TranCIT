"""
Granger Causality (GC) implementation.

This module provides the implementation of Granger Causality
calculation using the new modular architecture.
"""

import logging
from typing import Dict, Tuple

import numpy as np

from trancit.core.base import BaseAnalyzer, BaseResult
from trancit.core.exceptions import ComputationError, ValidationError
from trancit.utils.helpers import estimate_coefficients

logger = logging.getLogger(__name__)


class GrangerCausalityResult(BaseResult):
    """Result container for Granger Causality analysis."""

    def __init__(
        self,
        granger_causality: np.ndarray,
        coefficients: np.ndarray,
        residual_variances: np.ndarray,
    ):
        """
        Initialize Granger Causality result.

        Parameters
        ----------
        granger_causality : np.ndarray
            Granger Causality values
        coefficients : np.ndarray
            VAR coefficients
        residual_variances : np.ndarray
            Residual variances for each variable
        """
        super().__init__(
            granger_causality=granger_causality,
            coefficients=coefficients,
            residual_variances=residual_variances,
        )


class GrangerCausalityCalculator(BaseAnalyzer):
    """
    Granger Causality (GC) calculator.

    This class implements the Granger Causality algorithm for detecting
    linear causal relationships in time series data.
    """

    def __init__(
        self,
        model_order: int,
        time_mode: str = "inhomo",
        use_diagonal_covariance: bool = False,
        **kwargs,
    ):
        """
        Initialize Granger Causality calculator.

        Parameters
        ----------
        model_order : int
            Model order for VAR analysis
        time_mode : str
            Time mode: 'inhomo' (time-inhomogeneous) or 'homo' (homogeneous)
        use_diagonal_covariance : bool
            Whether to use diagonal covariance approximation
        **kwargs
            Additional configuration parameters
        """
        super().__init__(
            model_order=model_order,
            time_mode=time_mode,
            use_diagonal_covariance=use_diagonal_covariance,
            **kwargs,
        )

    def analyze(self, data: np.ndarray, **kwargs) -> GrangerCausalityResult:
        """
        Perform Granger Causality analysis on the given data.

        Parameters
        ----------
        data : np.ndarray
            Time series data with shape (n_vars, n_observations, n_trials)
        **kwargs
            Additional parameters

        Returns
        -------
        GrangerCausalityResult
            Granger Causality analysis results
        """
        self._validate_input_data(data)
        self._log_analysis_start(data.shape)

        try:
            current_data, lagged_data, n_time_steps = self._prepare_data_structures(
                data
            )

            result_arrays = self._initialize_result_arrays(
                n_time_steps, data.shape[1], data.shape[0]
            )

            if self.config["time_mode"] == "inhomo":
                self._compute_inhomogeneous_gc(
                    current_data,
                    lagged_data,
                    result_arrays,
                    data.shape[2],
                    data.shape[0],
                )
                self._adjust_outputs_for_inhomo(result_arrays)
            else:
                self._compute_homogeneous_gc(
                    current_data,
                    lagged_data,
                    result_arrays,
                    data.shape[2],
                    data.shape[0],
                )

            self._log_analysis_complete()

            return GrangerCausalityResult(
                granger_causality=result_arrays["granger_causality"],
                coefficients=result_arrays["coefficients"],
                residual_variances=result_arrays["residual_variances"],
            )

        except Exception as e:
            logger.error(f"Granger Causality analysis failed: {e}")
            raise ComputationError(
                f"Granger Causality analysis failed: {e}", "gc_computation", data.shape
            )

    def _validate_config(self) -> None:
        """Validate configuration parameters."""
        if self.config["model_order"] <= 0:
            raise ValidationError(
                "model_order must be positive",
                "model_order",
                self.config["model_order"],
            )

        if self.config["time_mode"] not in ["inhomo", "homo"]:
            raise ValidationError(
                "time_mode must be 'inhomo' or 'homo'",
                "time_mode",
                self.config["time_mode"],
            )

    def _validate_input_data(self, data: np.ndarray) -> None:
        """Validate input data format and dimensions."""
        super()._validate_input_data(data)

        if data.ndim != 3:
            raise ValidationError(
                "Input data must be 3D (n_vars, n_observations, n_trials)",
                "data_ndim",
                data.ndim,
            )

        if data.shape[0] != 2:
            raise ValidationError(
                "Input data must be bivariate (n_vars=2)", "n_vars", data.shape[0]
            )

    def _prepare_data_structures(
        self, time_series_data: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, int]:
        """
        Prepare data structures for analysis.

        Parameters
        ----------
        time_series_data : np.ndarray
            Input time series data

        Returns
        -------
        Tuple[np.ndarray, np.ndarray, int]
            current_data, lagged_data, n_time_steps
        """
        nvar, nobs, ntrials = time_series_data.shape
        r1 = self.config["model_order"] + 1

        extended_data = np.zeros((nvar, r1, nobs + r1 - 1, ntrials))
        for k in range(r1):
            extended_data[:, k, k : k + nobs, :] = time_series_data

        current_data = extended_data[:, 0, r1 - 1 : nobs, :]
        lagged_data = extended_data[
            :, 1 : self.config["model_order"] + 1, r1 - 1 : nobs - 1, :
        ]
        n_time_steps = current_data.shape[1] - 1

        return current_data, lagged_data, n_time_steps

    def _initialize_result_arrays(
        self, n_time_steps: int, nobs: int, nvar: int
    ) -> Dict[str, np.ndarray]:
        """
        Initialize arrays for storing results.

        Parameters
        ----------
        n_time_steps : int
            Number of time steps
        nobs : int
            Number of observations
        nvar : int
            Number of variables

        Returns
        -------
        Dict[str, np.ndarray]
            Dictionary of initialized arrays
        """
        return {
            "granger_causality": np.zeros((n_time_steps, 2)),
            "coefficients": np.full(
                (nobs, nvar, nvar * self.config["model_order"] + 1), np.nan
            ),
            "residual_variances": np.zeros((n_time_steps, 2)),
        }

    def _compute_inhomogeneous_gc(
        self,
        current_data: np.ndarray,
        lagged_data: np.ndarray,
        result_arrays: Dict[str, np.ndarray],
        ntrials: int,
        nvar: int,
    ) -> None:
        """Compute inhomogeneous Granger Causality."""
        n_time_steps = current_data.shape[1] - 1

        for t in range(n_time_steps):
            try:
                coeff, residual_cov = estimate_coefficients(
                    current_data[:, t, :].T,
                    lagged_data[:, : self.config["model_order"], t, :]
                    .reshape(nvar * self.config["model_order"], ntrials)
                    .T,
                    ntrials,
                )

                result_arrays["coefficients"][t, :, :] = coeff

                sigy = residual_cov[0, 0] + np.finfo(float).eps
                sigx = residual_cov[1, 1] + np.finfo(float).eps
                result_arrays["residual_variances"][t, 0] = sigx
                result_arrays["residual_variances"][t, 1] = sigy

                lagged_x_only = lagged_data[1, :, t, :].T
                current_x = current_data[1, t, :]
                _, sigx_reduced = estimate_coefficients(
                    current_x, lagged_x_only, ntrials
                )

                lagged_y_only = lagged_data[0, :, t, :].T
                current_y = current_data[0, t, :]
                _, sigy_reduced = estimate_coefficients(
                    current_y, lagged_y_only, ntrials
                )

                result_arrays["granger_causality"][t, 1] = np.log(
                    sigy_reduced / sigy
                )  # X -> Y
                result_arrays["granger_causality"][t, 0] = np.log(
                    sigx_reduced / sigx
                )  # Y -> X

            except Exception as e:
                logger.error(
                    f"Granger Causality computation failed at time step {t}: {e}"
                )
                result_arrays["granger_causality"][t] = np.zeros(2)
                result_arrays["residual_variances"][t] = np.zeros(2)

    def _compute_homogeneous_gc(
        self,
        current_data: np.ndarray,
        lagged_data: np.ndarray,
        result_arrays: Dict[str, np.ndarray],
        ntrials: int,
    ) -> None:
        """Compute homogeneous Granger Causality."""
        coeff, residual_cov = estimate_coefficients(current_data, lagged_data, ntrials)

        for t in range(result_arrays["coefficients"].shape[0]):
            result_arrays["coefficients"][t] = coeff.T

        sigy = residual_cov[0, 0] + np.finfo(float).eps
        sigx = residual_cov[1, 1] + np.finfo(float).eps
        result_arrays["residual_variances"][0, 0] = sigx
        result_arrays["residual_variances"][0, 1] = sigy

        lagged_x_only = (
            lagged_data[1, :, :, :].reshape(self.config["model_order"], -1).T
        )
        lagged_y_only = (
            lagged_data[0, :, :, :].reshape(self.config["model_order"], -1).T
        )
        current_x = current_data[1, :, :].flatten()
        current_y = current_data[0, :, :].flatten()

        _, sigx_reduced = estimate_coefficients(current_x, lagged_x_only, ntrials)
        _, sigy_reduced = estimate_coefficients(current_y, lagged_y_only, ntrials)

        result_arrays["granger_causality"][0, 1] = np.log(sigy_reduced / sigy)  # X -> Y
        result_arrays["granger_causality"][0, 0] = np.log(sigx_reduced / sigx)  # Y -> X

    def _adjust_outputs_for_inhomo(self, result_arrays: Dict[str, np.ndarray]) -> None:
        """Adjust outputs for inhomogeneous mode by adding NaN blocks."""
        if self.config["time_mode"] == "inhomo":
            nan_block = np.full((self.config["model_order"], 2), np.nan)
            result_arrays["granger_causality"] = np.vstack(
                [nan_block, result_arrays["granger_causality"]]
            )
            result_arrays["residual_variances"] = np.vstack(
                [nan_block, result_arrays["residual_variances"]]
            )
