"""
Dynamic Causal Strength (DCS) implementation.

This module provides the implementation of Dynamic Causal Strength
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


class DCSResult(BaseResult):
    """Result container for DCS analysis."""

    def __init__(
        self,
        causal_strength: np.ndarray,
        transfer_entropy: np.ndarray,
        granger_causality: np.ndarray,
        coefficients: np.ndarray,
        te_residual_cov: np.ndarray,
    ):
        """
        Initialize DCS result.

        Parameters
        ----------
        causal_strength : np.ndarray
            Dynamic Causal Strength values
        transfer_entropy : np.ndarray
            Transfer Entropy values
        granger_causality : np.ndarray
            Granger Causality values
        coefficients : np.ndarray
            VAR coefficients
        te_residual_cov : np.ndarray
            Transfer entropy residual covariance
        """
        super().__init__(
            causal_strength=causal_strength,
            transfer_entropy=transfer_entropy,
            granger_causality=granger_causality,
            coefficients=coefficients,
            te_residual_cov=te_residual_cov,
        )


class DCSCalculator(BaseAnalyzer):
    """
    Dynamic Causal Strength (DCS) calculator.

    This class implements the DCS algorithm for quantifying
    causal relationships in time series data.
    """

    def __init__(
        self,
        model_order: int,
        time_mode: str = "inhomo",
        use_diagonal_covariance: bool = False,
        **kwargs,
    ):
        """
        Initialize DCS calculator.

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

    def analyze(self, data: np.ndarray, **kwargs) -> DCSResult:
        """
        Perform DCS analysis on the given data.

        Parameters
        ----------
        data : np.ndarray
            Time series data with shape (n_vars, n_observations, n_trials)
        **kwargs
            Additional parameters

        Returns
        -------
        DCSResult
            DCS analysis results
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
                self._compute_inhomogeneous_causality(
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
                self._compute_homogeneous_causality(
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

            return DCSResult(
                causal_strength=result_arrays["causal_strength"],
                transfer_entropy=result_arrays["transfer_entropy"],
                granger_causality=result_arrays["granger_causality"],
                coefficients=result_arrays["coefficients"],
                te_residual_cov=result_arrays["te_residual_cov"],
            )

        except Exception as e:
            logger.error(f"DCS analysis failed: {e}")
            raise ComputationError(
                f"DCS analysis failed: {e}", "dcs_computation", data.shape
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
        Prepare data structures for analysis exactly like the previous implementation.

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
        Initialize arrays for storing results exactly like the previous implementation.

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
            "causal_strength": np.zeros((n_time_steps, 2)),
            "granger_causality": np.zeros((n_time_steps, 2)),
            "coefficients": np.full(
                (nobs, nvar, nvar * self.config["model_order"] + 1), np.nan
            ),
        }

    def _compute_inhomogeneous_causality(
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
        """Compute inhomogeneous causality exactly like the previous implementation."""
        n_time_steps = current_data.shape[1] - 1

        for t in range(n_time_steps):
            try:
                logging.debug(f"Processing time step {t}/{n_time_steps - 1}")
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
                sigy = residual_cov[0, 0] or np.finfo(float).eps
                sigx = residual_cov[1, 1] or np.finfo(float).eps

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

                if not self.config["use_diagonal_covariance"]:
                    result_arrays["causal_strength"][t, 1] = 0.5 * np.log(
                        (sigy + b.T @ cov_xp_reg @ b) / sigy
                    )
                    result_arrays["causal_strength"][t, 0] = 0.5 * np.log(
                        (sigx + c.T @ cov_yp_reg @ c) / sigx
                    )
                else:
                    result_arrays["causal_strength"][t, 1] = 0.5 * np.log(
                        (sigy + b.T @ np.diag(np.diag(cov_xp_reg)) @ b) / sigy
                    )
                    result_arrays["causal_strength"][t, 0] = 0.5 * np.log(
                        (sigx + c.T @ np.diag(np.diag(cov_yp_reg)) @ c) / sigx
                    )

                lagged_x_p = lagged_data[1, :, t, :].T
                lagged_y_p = lagged_data[0, :, t, :].T
                current_x_t = current_data[1, t, :]
                current_y_t = current_data[0, t, :]
                _, sigx_r = estimate_coefficients(current_x_t, lagged_x_p, ntrials)
                _, sigy_r = estimate_coefficients(current_y_t, lagged_y_p, ntrials)
                result_arrays["granger_causality"][t, 1] = np.log(sigy_r / sigy)
                result_arrays["granger_causality"][t, 0] = np.log(sigx_r / sigx)
            except Exception as e:
                logger.error(f"Computation failed at time step {t}: {e}")
                result_arrays["causal_strength"][t] = np.zeros(2)
                result_arrays["transfer_entropy"][t] = np.zeros(2)
                result_arrays["granger_causality"][t] = np.zeros(2)

    def _compute_homogeneous_causality(
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
        """Compute homogeneous causality exactly like the previous implementation."""
        current_data.shape[1] - 1

        coeff, residual_cov = estimate_coefficients(current_data, lagged_data, ntrials)

        for t in range(result_arrays["coefficients"].shape[0]):
            result_arrays["coefficients"][t] = coeff.T

        a_square = coeff[:, :-1].reshape(nvar, nvar, self.config["model_order"])
        b = a_square[0, 1, :]  # X -> Y
        c = a_square[1, 0, :]  # Y -> X
        sigy = residual_cov[0, 0] or np.finfo(float).eps
        sigx = residual_cov[1, 1] or np.finfo(float).eps

        cov_xp_avg = np.mean(cov_xp, axis=0)
        cov_yp_avg = np.mean(cov_yp, axis=0)
        cov_xy_p_avg = np.mean(c_xyp, axis=0)
        cov_yx_p_avg = np.mean(c_yxp, axis=0)

        cov_yp_reg = regularize_if_singular(cov_yp_avg)
        cov_xp_reg = regularize_if_singular(cov_xp_avg)

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

        if not self.config["use_diagonal_covariance"]:
            result_arrays["causal_strength"][0, 1] = 0.5 * np.log(
                (sigy + b.T @ cov_xp_reg @ b) / sigy
            )
            result_arrays["causal_strength"][0, 0] = 0.5 * np.log(
                (sigx + c.T @ cov_yp_reg @ c) / sigx
            )
        else:
            result_arrays["causal_strength"][0, 1] = 0.5 * np.log(
                (sigy + b.T @ np.diag(np.diag(cov_xp_reg)) @ b) / sigy
            )
            result_arrays["causal_strength"][0, 0] = 0.5 * np.log(
                (sigx + c.T @ np.diag(np.diag(cov_yp_reg)) @ c) / sigx
            )

        lagged_x_p = lagged_data[1, :, :, :].reshape(self.config["model_order"], -1).T
        lagged_y_p = lagged_data[0, :, :, :].reshape(self.config["model_order"], -1).T
        current_x_t = current_data[1, :, :].flatten()
        current_y_t = current_data[0, :, :].flatten()
        _, sigx_r = estimate_coefficients(current_x_t, lagged_x_p, ntrials)
        _, sigy_r = estimate_coefficients(current_y_t, lagged_y_p, ntrials)
        result_arrays["granger_causality"][0, 1] = np.log(sigy_r / sigy)
        result_arrays["granger_causality"][0, 0] = np.log(sigx_r / sigx)

    def _adjust_outputs_for_inhomo(self, result_arrays: Dict[str, np.ndarray]) -> None:
        """Adjust outputs for inhomogeneous mode by adding NaN blocks exactly
        like the previous implementation."""
        if self.config["time_mode"] == "inhomo":
            nan_block = np.full((self.config["model_order"], 2), np.nan)
            result_arrays["granger_causality"] = np.vstack(
                [nan_block, result_arrays["granger_causality"]]
            )
            result_arrays["transfer_entropy"] = np.vstack(
                [nan_block, result_arrays["transfer_entropy"]]
            )
            result_arrays["causal_strength"] = np.vstack(
                [nan_block, result_arrays["causal_strength"]]
            )
