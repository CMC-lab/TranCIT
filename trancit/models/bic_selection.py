"""
BIC (Bayesian Information Criterion) model selection.

This module provides BIC-based model order selection for VAR models.
"""

import logging
from typing import Dict, Tuple

import numpy as np

from trancit import PipelineConfig
from trancit.utils import compute_event_statistics

from ..core.exceptions import ValidationError

logger = logging.getLogger(__name__)


class BICSelector:
    """
    Bayesian Information Criterion (BIC) selector for VAR model order selection.

    This class provides methods for computing BIC values and selecting optimal
    model orders for Vector Autoregression (VAR) models.
    """

    def __init__(self, config: PipelineConfig):
        """
        Initialize BIC selector.

        Parameters
        ----------
        config : PipelineConfig
            Pipeline configuration
        """
        self.config = config
        self._validate_parameters()

    def _validate_parameters(self) -> None:
        """Validate initialization parameters."""
        if self.config.bic.momax <= 0:
            raise ValidationError(
                "max_order must be positive", "max_order", self.config.bic.momax
            )
        if self.config.bic.mode not in ["biased", "debiased"]:
            raise ValidationError(
                "mode must be 'biased' or 'debiased'", "mode", self.config.bic.mode
            )

    def _compute_multi_trial_bic(self, event_data_max_order: np.ndarray) -> Dict:
        """
        Calculate Bayesian Information Criterion (BIC) for multiple model orders across trial data.

        This function computes BIC values for Vector Autoregression (VAR) models
        of orders 1 to momax,
        supporting model selection with multiple BIC variants.

        Parameters
        ----------
        event_data_max_order : np.ndarray
            Time series data for the maximum model order,
            shape (n_vars * (max_order + 1), n_observations, n_trials).
        bic_params : dict
            Parameters for BIC calculation, containing:
                - "Params": Sub-dict with "BIC": {"momax": int}, the maximum model order.

        Returns
        -------
        dict
            Dictionary containing:
                - 'bic': BIC values for each model order and variant, shape (max_order, 4).
                - 'penalty_terms': Penalty terms for BIC, shape (max_order, 4).
                - 'log_likelihood': Log-likelihood for each model order, shape (max_order,).
                - 'sum_log_det_hessian': Sum of log determinant of Hessian, shape (max_order,).
                - 'optimal_orders': Optimal model orders for each BIC variant, shape (4,).

        Raises
        ------
        ValueError
            If input data shape is inconsistent with expected dimensions.
        """
        max_order = self.config.bic.momax
        print(f"max_order: {max_order}")
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

            log_likelihood, sum_log_det_hessian = self._compute_bic_for_model(
                data_subset, model_order, self.config
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

    def _compute_bic_for_model(
        self, event_data: np.ndarray, model_order: int, config: PipelineConfig
    ) -> Tuple[float, float]:
        """
        Compute log-likelihood and sum of log determinant of Hessian for a specific model order.

        Supports 'biased' mode currently; 'debiased' mode is planned for future implementation.

        Parameters
        ----------
        event_data : np.ndarray
            Event data for the model order,
            shape (n_vars * (model_order + 1), n_observations, n_trials).
        model_order : int
            The VAR model order to evaluate.
        bic_params : dict
            BIC parameters including:
                - 'Params': Sub-dict with 'BIC': {'mode': str}, e.g., 'biased'.
                - 'EstimMode': Estimation mode, either 'OLS' or 'RLS'.

        Returns
        -------
        Tuple[float, float]
            - log_likelihood: The log-likelihood of the model.
            - sum_log_det_hessian: Sum of the log determinant of the Hessian.

        Raises
        ------
        ValueError
            If 'mode' or 'EstimMode' is invalid.
        """
        total_var_lag, n_observations, n_trials = event_data.shape
        n_vars = total_var_lag // (model_order + 1)

        mode = config.bic.mode
        if mode not in ["biased"]:
            raise ValueError(
                f"Unsupported mode '{mode}'; only 'biased' is implemented."
            )

        if mode == "biased":
            stats = compute_event_statistics(event_data, model_order)
        else:
            raise NotImplementedError("'debiased' mode is not yet supported.")

        log_det_hessian = np.zeros(n_observations)
        residual_determinants = np.zeros(n_observations)

        estim_mode = config.bic.estim_mode
        if estim_mode not in ["OLS", "RLS"]:
            raise ValueError(
                f"Invalid EstimMode '{estim_mode}'; must be 'OLS' or 'RLS'."
            )

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
            f"Model order {model_order}: log_likelihood={log_likelihood:.4f}, "
            f"sum_log_det_hessian={sum_log_det_hessian:.4f}"
        )

        return log_likelihood, sum_log_det_hessian
