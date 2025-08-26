"""
Transfer Entropy (TE) implementation.

This module provides the implementation of Transfer Entropy
calculation using the new modular architecture.
"""

import logging
from typing import Dict, Tuple

import numpy as np

from trancit.core.base import BaseAnalyzer, BaseResult
from trancit.core.exceptions import ComputationError, ValidationError
from trancit.utils.helpers import compute_covariances, estimate_coefficients
from trancit.utils.preprocess import regularize_if_singular

logger = logging.getLogger(__name__)


class TransferEntropyResult(BaseResult):
    """Result container for Transfer Entropy analysis."""

    def __init__(
        self,
        transfer_entropy: np.ndarray,
        te_residual_cov: np.ndarray,
        coefficients: np.ndarray,
    ):
        """
        Initialize Transfer Entropy result.

        Parameters
        ----------
        transfer_entropy : np.ndarray
            Transfer Entropy values
        te_residual_cov : np.ndarray
            Transfer entropy residual covariance
        coefficients : np.ndarray
            VAR coefficients
        """
        super().__init__(
            transfer_entropy=transfer_entropy,
            te_residual_cov=te_residual_cov,
            coefficients=coefficients,
        )


class TransferEntropyCalculator(BaseAnalyzer):
    """
    Transfer Entropy (TE) calculator.

    This class implements the Transfer Entropy algorithm for quantifying
    information flow between time series variables.
    """

    def __init__(
        self,
        model_order: int,
        time_mode: str = "inhomo",
        use_diagonal_covariance: bool = False,
        **kwargs,
    ):
        """
        Initialize Transfer Entropy calculator.

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
        # VAREstimator is not used in this implementation - using direct computation

    def analyze(self, data: np.ndarray, **kwargs) -> TransferEntropyResult:
        """
        Perform Transfer Entropy analysis on the given data.

        Parameters
        ----------
        data : np.ndarray
            Time series data with shape (n_vars, n_observations, n_trials)
        **kwargs
            Additional parameters

        Returns
        -------
        TransferEntropyResult
            Transfer Entropy analysis results
        """
        self._validate_input_data(data)
        self._log_analysis_start(data.shape)

        try:
            current_data, lagged_data, n_time_steps = self._prepare_data_structures(
                data
            )

            cov_xp, cov_yp, c_xyp, c_yxp = compute_covariances(
                lagged_data, n_time_steps, self.config["model_order"]
            )

            result_arrays = self._initialize_result_arrays(
                n_time_steps, data.shape[1], data.shape[0]
            )

            if self.config["time_mode"] == "inhomo":
                self._compute_inhomogeneous_te(
                    current_data,
                    lagged_data,
                    cov_xp,
                    cov_yp,
                    c_xyp,
                    c_yxp,
                    result_arrays,
                    data.shape[2],
                    data.shape[0],
                )
                self._adjust_outputs_for_inhomo(result_arrays)
            else:
                self._compute_homogeneous_te(
                    current_data,
                    lagged_data,
                    cov_xp,
                    cov_yp,
                    c_xyp,
                    c_yxp,
                    result_arrays,
                    data.shape[2],
                    data.shape[0],
                )

            self._log_analysis_complete()

            return TransferEntropyResult(
                transfer_entropy=result_arrays["transfer_entropy"],
                te_residual_cov=result_arrays["te_residual_cov"],
                coefficients=result_arrays["coefficients"],
            )

        except Exception as e:
            logger.error(f"Transfer Entropy analysis failed: {e}")
            raise ComputationError(
                f"Transfer Entropy analysis failed: {e}", "te_computation", data.shape
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
            "te_residual_cov": np.zeros((n_time_steps, 2)),
            "transfer_entropy": np.zeros((n_time_steps, 2)),
            "coefficients": np.full(
                (nobs, nvar, nvar * self.config["model_order"] + 1), np.nan
            ),
        }

    def _compute_inhomogeneous_te(
        self,
        current_data: np.ndarray,
        lagged_data: np.ndarray,
        cov_xp: np.ndarray,
        cov_yp: np.ndarray,
        c_xyp: np.ndarray,
        c_yxp: np.ndarray,
        result_arrays: Dict[str, np.ndarray],
        ntrials: int,
        nvar: int,
    ) -> None:
        """Compute inhomogeneous Transfer Entropy."""
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

                a_square = coeff[:, :-1].reshape(nvar, nvar, self.config["model_order"])
                b = a_square[0, 1, :]  # X -> Y
                c = a_square[1, 0, :]  # Y -> X
                sigy = residual_cov[0, 0] + np.finfo(float).eps
                sigx = residual_cov[1, 1] + np.finfo(float).eps

                cov_yp_reg = regularize_if_singular(cov_yp[t])
                cov_xp_reg = regularize_if_singular(cov_xp[t])

                result_arrays["te_residual_cov"][t, 1] = (
                    sigy
                    + b.T @ cov_xp_reg @ b
                    - b.T @ c_xyp[t] @ np.linalg.inv(cov_yp_reg) @ c_xyp[t].T @ b
                )
                result_arrays["te_residual_cov"][t, 0] = (
                    sigx
                    + c.T @ cov_yp_reg @ c
                    - c.T @ c_yxp[t] @ np.linalg.inv(cov_xp_reg) @ c_yxp[t].T @ c
                )

                result_arrays["transfer_entropy"][t, 1] = 0.5 * np.log(
                    result_arrays["te_residual_cov"][t, 1] / sigy
                )
                result_arrays["transfer_entropy"][t, 0] = 0.5 * np.log(
                    result_arrays["te_residual_cov"][t, 0] / sigx
                )

            except Exception as e:
                logger.error(
                    f"Transfer Entropy computation failed at time step {t}: {e}"
                )
                result_arrays["transfer_entropy"][t] = np.zeros(2)
                result_arrays["te_residual_cov"][t] = np.zeros(2)

    def _compute_homogeneous_te(
        self,
        current_data: np.ndarray,
        lagged_data: np.ndarray,
        cov_xp: np.ndarray,
        cov_yp: np.ndarray,
        c_xyp: np.ndarray,
        c_yxp: np.ndarray,
        result_arrays: Dict[str, np.ndarray],
        ntrials: int,
        nvar: int,
    ) -> None:
        """Compute homogeneous Transfer Entropy."""
        coeff, residual_cov = estimate_coefficients(current_data, lagged_data, ntrials)

        for t in range(result_arrays["coefficients"].shape[0]):
            result_arrays["coefficients"][t] = coeff.T

        a_square = coeff[:, :-1].reshape(nvar, nvar, self.config["model_order"])
        b = a_square[0, 1, :]  # X -> Y
        c = a_square[1, 0, :]  # Y -> X
        sigy = residual_cov[0, 0] + np.finfo(float).eps
        sigx = residual_cov[1, 1] + np.finfo(float).eps

        cov_xp_avg = np.mean(cov_xp, axis=0)
        cov_yp_avg = np.mean(cov_yp, axis=0)
        cov_xy_p_avg = np.mean(c_xyp, axis=0)
        cov_yx_p_avg = np.mean(c_yxp, axis=0)

        result_arrays["transfer_entropy"][0, 1] = 0.5 * np.log(
            (
                sigy
                + b.T @ cov_xp_avg @ b
                - b.T @ cov_xy_p_avg @ np.linalg.inv(cov_yp_avg) @ cov_xy_p_avg.T @ b
            )
            / sigy
        )
        result_arrays["transfer_entropy"][0, 0] = 0.5 * np.log(
            (
                sigx
                + c.T @ cov_yp_avg @ c
                - c.T @ cov_yx_p_avg @ np.linalg.inv(cov_xp_avg) @ cov_yx_p_avg.T @ c
            )
            / sigx
        )

    def _adjust_outputs_for_inhomo(self, result_arrays: Dict[str, np.ndarray]) -> None:
        """Adjust outputs for inhomogeneous mode by adding NaN blocks."""
        if self.config["time_mode"] == "inhomo":
            nan_block = np.full((self.config["model_order"], 2), np.nan)
            result_arrays["transfer_entropy"] = np.vstack(
                [nan_block, result_arrays["transfer_entropy"]]
            )
            result_arrays["te_residual_cov"] = np.vstack(
                [nan_block, result_arrays["te_residual_cov"]]
            )
