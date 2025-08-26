"""
Base classes for causality analysis.

This module provides the foundation for all causality calculations
with common interfaces and shared functionality.
"""

from abc import ABC, abstractmethod
from typing import Dict, Tuple

import numpy as np


class CausalityResult:
    """Container for causality analysis results."""

    def __init__(
        self,
        causal_strength: np.ndarray,
        transfer_entropy: np.ndarray,
        granger_causality: np.ndarray,
        coefficients: np.ndarray,
        te_residual_cov: np.ndarray,
    ):
        self.causal_strength = causal_strength
        self.transfer_entropy = transfer_entropy
        self.granger_causality = granger_causality
        self.coefficients = coefficients
        self.te_residual_cov = te_residual_cov

    def __repr__(self) -> str:
        return (
            f"CausalityResult("
            f"causal_strength={self.causal_strength.shape}, "
            f"transfer_entropy={self.transfer_entropy.shape}, "
            f"granger_causality={self.granger_causality.shape})"
        )


class BaseCausalityAnalyzer(ABC):
    """
    Abstract base class for causality analysis.

    This class defines the common interface and shared functionality
    for all causality calculation methods.
    """

    def __init__(self, model_order: int, time_mode: str = "inhomo"):
        """
        Initialize the causality analyzer.

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
            raise ValueError("model_order must be positive")
        if self.time_mode not in ["inhomo", "homo"]:
            raise ValueError("time_mode must be 'inhomo' or 'homo'")

    @abstractmethod
    def compute(self, time_series_data: np.ndarray, **kwargs) -> CausalityResult:
        """
        Compute causality measures.

        Parameters
        ----------
        time_series_data : np.ndarray
            Input time series data
        **kwargs
            Additional parameters specific to each method

        Returns
        -------
        CausalityResult
            Results of the causality analysis
        """

    def _validate_input_data(self, data: np.ndarray) -> None:
        """Validate input data format and dimensions."""
        if not isinstance(data, np.ndarray):
            raise TypeError("Input data must be a NumPy array")

        if data.ndim != 3:
            raise ValueError("Input data must be 3D (nvar, nobs, ntrials)")

        if data.shape[0] != 2:
            raise ValueError("Input data must be bivariate (nvar=2)")

        if np.any(np.isnan(data)) or np.any(np.isinf(data)):
            raise ValueError("Input data contains NaN or Inf values")

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
        r1 = self.model_order + 1

        extended_data = np.zeros((nvar, r1, nobs + r1 - 1, ntrials))
        for k in range(r1):
            extended_data[:, k, k : k + nobs, :] = time_series_data

        current_data = extended_data[:, 0, r1 - 1 : nobs, :]
        lagged_data = extended_data[:, 1 : self.model_order + 1, r1 - 1 : nobs - 1, :]
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
            "causal_strength": np.zeros((n_time_steps, 2)),
            "granger_causality": np.zeros((n_time_steps, 2)),
            "coefficients": np.full((nobs, nvar, nvar * self.model_order + 1), np.nan),
        }

    def _adjust_outputs_for_inhomo(self, result_arrays: Dict[str, np.ndarray]) -> None:
        """
        Adjust outputs for inhomogeneous mode by adding NaN blocks.

        Parameters
        ----------
        result_arrays : Dict[str, np.ndarray]
            Arrays to adjust
        """
        if self.time_mode == "inhomo":
            nan_block = np.full((self.model_order, 2), np.nan)
            result_arrays["granger_causality"] = np.vstack(
                [nan_block, result_arrays["granger_causality"]]
            )
            result_arrays["transfer_entropy"] = np.vstack(
                [nan_block, result_arrays["transfer_entropy"]]
            )
            result_arrays["causal_strength"] = np.vstack(
                [nan_block, result_arrays["causal_strength"]]
            )
