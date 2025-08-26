"""
Relative Dynamic Causal Strength (rDCS) implementation.

This module provides the implementation of time-varying causality
including Transfer Entropy (TE), Dynamic Causal Strength (DCS),
and Relative Dynamic Causal Strength (rDCS).
"""

import logging
from typing import Dict

import numpy as np

from trancit.core.base import BaseAnalyzer, BaseResult
from trancit.core.exceptions import ComputationError, ValidationError
from trancit.utils.preprocess import regularize_if_singular

logger = logging.getLogger(__name__)


class RelativeDCSResult(BaseResult):
    """Result container for Relative Dynamic Causal Strength analysis."""

    def __init__(
        self,
        transfer_entropy: np.ndarray,
        dynamic_causal_strength: np.ndarray,
        relative_dynamic_causal_strength: np.ndarray,
        coefficients: np.ndarray,
    ):
        """
        Initialize Relative DCS result.

        Parameters
        ----------
        transfer_entropy : np.ndarray
            Transfer Entropy values
        dynamic_causal_strength : np.ndarray
            Dynamic Causal Strength values
        relative_dynamic_causal_strength : np.ndarray
            Relative Dynamic Causal Strength values
        coefficients : np.ndarray
            VAR coefficients
        """
        super().__init__(
            transfer_entropy=transfer_entropy,
            dynamic_causal_strength=dynamic_causal_strength,
            relative_dynamic_causal_strength=relative_dynamic_causal_strength,
            coefficients=coefficients,
        )


class RelativeDCSCalculator(BaseAnalyzer):
    """
    Relative Dynamic Causal Strength (rDCS) calculator.

    This class implements the Relative Dynamic Causal Strength algorithm
    for quantifying causal relationships relative to a baseline period.
    """

    def __init__(
        self,
        model_order: int,
        reference_time: int,
        estimation_mode: str = "OLS",
        use_diagonal_covariance: bool = False,
        use_old_version: bool = False,
        **kwargs,
    ):
        """
        Initialize Relative DCS calculator.

        Parameters
        ----------
        model_order : int
            Model order for VAR analysis
        reference_time : int
            Reference time point for baseline calculation
        estimation_mode : str
            Estimation mode: 'OLS' or 'RLS'
        use_diagonal_covariance : bool
            Whether to use diagonal covariance approximation
        use_old_version : bool
            Whether to use old version of rDCS calculation
        **kwargs
            Additional configuration parameters
        """
        super().__init__(
            model_order=model_order,
            reference_time=reference_time,
            estimation_mode=estimation_mode,
            use_diagonal_covariance=use_diagonal_covariance,
            use_old_version=use_old_version,
            **kwargs,
        )

    def analyze(  # type: ignore[override]
        self, event_data: np.ndarray, stats: Dict, **kwargs
    ) -> RelativeDCSResult:
        """
        Perform Relative Dynamic Causal Strength analysis.

        Parameters
        ----------
        event_data : np.ndarray
            Event data array of shape (nvar * (model_order + 1), nobs, ntrials)
        stats : Dict
            Model statistics with keys:
            - 'OLS' or 'RLS': Sub-dict with 'At' (coefficients) and 'Sigma_Et' (residual covariance)
            - 'Sigma': Covariance matrices
            - 'mean': Mean values
        **kwargs
            Additional parameters

        Returns
        -------
        RelativeDCSResult
            Relative DCS analysis results
        """
        self._validate_input_data(event_data, stats)
        self._log_analysis_start(event_data.shape)

        try:
            causal_params = {
                "ref_time": self.config["reference_time"],
                "estim_mode": self.config["estimation_mode"],
                "morder": self.config["model_order"],
                "diag_flag": self.config["use_diagonal_covariance"],
                "old_version": self.config["use_old_version"],
            }

            causality_measures = time_varying_causality(
                event_data, stats, causal_params
            )

            self._log_analysis_complete()

            return RelativeDCSResult(
                transfer_entropy=causality_measures["TE"],
                dynamic_causal_strength=causality_measures["DCS"],
                relative_dynamic_causal_strength=causality_measures["rDCS"],
                coefficients=stats[self.config["estimation_mode"]]["At"],
            )

        except Exception as e:
            logger.error(f"Relative DCS analysis failed: {e}")
            raise ComputationError(
                f"Relative DCS analysis failed: {e}",
                "rdcs_computation",
                event_data.shape,
            )

    def _validate_config(self) -> None:
        """Validate configuration parameters."""
        if self.config["model_order"] <= 0:
            raise ValidationError(
                "model_order must be positive",
                "model_order",
                self.config["model_order"],
            )

        if self.config["reference_time"] < 0:
            raise ValidationError(
                "reference_time must be non-negative",
                "reference_time",
                self.config["reference_time"],
            )

        if self.config["estimation_mode"] not in ["OLS", "RLS"]:
            raise ValidationError(
                "estimation_mode must be 'OLS' or 'RLS'",
                "estimation_mode",
                self.config["estimation_mode"],
            )

    def _validate_input_data(  # type: ignore[override]
        self, event_data: np.ndarray, stats: Dict
    ) -> None:
        """Validate input data format and dimensions."""
        super()._validate_input_data(event_data)

        if event_data.ndim != 3:
            raise ValidationError(
                "event_data must be 3D", "event_data_ndim", event_data.ndim
            )

        if not isinstance(stats, dict):
            raise ValidationError(
                "stats must be a dictionary", "stats_type", type(stats)
            )

        required_keys = ["OLS", "Sigma", "mean"]
        for key in required_keys:
            if key not in stats:
                raise ValidationError(
                    f"stats must contain '{key}'", "stats_missing_key", key
                )


def time_varying_causality(
    event_data: np.ndarray, stats: Dict, causal_params: Dict
) -> Dict[str, np.ndarray]:
    """
    Compute time-varying causality measures for bivariate signals.

    Calculates Transfer Entropy (TE), Dynamic Causal Strength (DCS), and Relative Dynamic Causal
    Strength (rDCS) based on a VAR model. This function maintains exact mathematical alignment
    with the previous implementation while providing enhanced error handling and validation.

    Parameters
    ----------
    event_data : np.ndarray
        Event data array of shape (nvar * (model_order + 1), nobs, ntrials).
        Must be bivariate (nvar = 2).
    stats : Dict
        Model statistics with keys:
        - 'OLS' or 'RLS': Sub-dict with 'At' (coefficients) and 'Sigma_Et' (residual covariance).
        - 'Sigma': Covariance matrices of shape
          (nobs, nvar * (model_order + 1), nvar * (model_order + 1)).
        - 'mean': Mean values of shape (nvar * (model_order + 1), nobs).
    causal_params : Dict
        Parameters with keys:
        - 'ref_time': Reference time index for rDCS calculation.
        - 'estim_mode': 'OLS' or 'RLS' estimation mode.
        - 'morder': Model order (number of lags).
        - 'diag_flag': Boolean for diagonal covariance approximation.
        - 'old_version': Boolean for rDCS calculation method.

    Returns
    -------
    Dict[str, np.ndarray]
        Causality measures:
        - 'TE': Transfer Entropy, shape (nobs, 2) where [:, 0] is Y->X, [:, 1] is X->Y.
        - 'DCS': Dynamic Causal Strength, shape (nobs, 2) where [:, 0] is Y->X, [:, 1] is X->Y.
        - 'rDCS': Relative Dynamic Causal Strength, shape (nobs, 2)
          where [:, 0] is Y->X, [:, 1] is X->Y.

    Raises
    ------
    ValidationError
        If input data or parameters are invalid.
    ComputationError
        If computation fails due to numerical issues.

    Notes
    -----
    - The function assumes bivariate data (nvar = 2).
    - Transfer Entropy (TE) measures directed information flow.
    - Dynamic Causal Strength (DCS) measures direct causal influence.
    - Relative Dynamic Causal Strength (rDCS) measures causal influence relative to a baseline.
    - All measures are computed using the Structural Causal Model (SCM) framework.

    Examples
    --------
    >>> event_data = np.random.randn(6, 50, 10)  # (nvar * (morder + 1), nobs, ntrials)
    >>> stats = {
    ...     'OLS': {
    ...         'At': np.random.randn(50, 2, 4),
    ...         'Sigma_Et': np.array([np.eye(2) for _ in range(50)])
    ...     },
    ...     'Sigma': np.random.randn(50, 6, 6),
    ...     'mean': np.random.randn(6, 50)
    ... }
    >>> causal_params = {
    ...     'ref_time': 10, 'estim_mode': 'OLS', 'morder': 2,
    ...     'diag_flag': False, 'old_version': False
    ... }
    >>> result = time_varying_causality(event_data, stats, causal_params)
    >>> print(f"TE shape: {result['TE'].shape}")
    >>> print(f"DCS shape: {result['DCS'].shape}")
    >>> print(f"rDCS shape: {result['rDCS'].shape}")
    """
    _validate_time_varying_causality_inputs(event_data, stats, causal_params)

    _, nobs, ntrials = event_data.shape
    nvar = stats["OLS"]["At"].shape[1]
    ref_time = _normalize_ref_time(causal_params["ref_time"], nobs)
    estim_mode = causal_params["estim_mode"]
    morder = causal_params["morder"]
    diag_flag = causal_params["diag_flag"]
    old_version = causal_params["old_version"]

    logger.info(
        f"Computing time-varying causality: mode={estim_mode}, morder={morder}, ref_time={ref_time}"
    )

    causality_measures = {
        "TE": np.zeros((nobs, 2)),
        "DCS": np.zeros((nobs, 2)),
        "rDCS": np.zeros((nobs, 2)),
    }

    for t in range(nobs):
        try:
            _compute_causality_at_timepoint(
                t,
                event_data,
                stats,
                causality_measures,
                nvar,
                ntrials,
                ref_time,
                estim_mode,
                morder,
                diag_flag,
                old_version,
            )
        except Exception as e:
            logger.error(f"Computation failed at time point {t}: {e}")
            causality_measures["TE"][t] = np.zeros(2)
            causality_measures["DCS"][t] = np.zeros(2)
            causality_measures["rDCS"][t] = np.zeros(2)

    logger.info("Time-varying causality computation completed")
    return causality_measures


def _validate_time_varying_causality_inputs(
    event_data: np.ndarray, stats: Dict, causal_params: Dict
) -> None:
    """
    Validate inputs for time_varying_causality function.

    Parameters
    ----------
    event_data : np.ndarray
        Event data array
    stats : Dict
        Model statistics
    causal_params : Dict
        Causality parameters

    Raises
    ------
    ValidationError
        If inputs are invalid
    """
    if not isinstance(event_data, np.ndarray):
        raise ValidationError(
            "event_data must be a NumPy array", "event_data_type", type(event_data)
        )

    if event_data.ndim != 3:
        raise ValidationError(
            "event_data must be 3D", "event_data_ndim", event_data.ndim
        )

    if not isinstance(stats, dict):
        raise ValidationError("stats must be a dictionary", "stats_type", type(stats))

    required_keys = ["OLS", "Sigma", "mean"]
    for key in required_keys:
        if key not in stats:
            raise ValidationError(
                f"stats must contain '{key}'", "stats_missing_key", key
            )

    if not isinstance(causal_params, dict):
        raise ValidationError(
            "causal_params must be a dictionary",
            "causal_params_type",
            type(causal_params),
        )

    required_params = ["ref_time", "estim_mode", "morder", "diag_flag", "old_version"]
    for param in required_params:
        if param not in causal_params:
            raise ValidationError(
                f"causal_params must contain '{param}'", "causal_params_missing", param
            )

    if causal_params["estim_mode"] not in ["OLS", "RLS"]:
        raise ValidationError(
            "estim_mode must be 'OLS' or 'RLS'",
            "estim_mode",
            causal_params["estim_mode"],
        )

    _validate_ref_time(causal_params["ref_time"])


def _validate_ref_time(ref_time: object) -> None:
    """
    Validate ref_time parameter.

    Parameters
    ----------
    ref_time : object
        Reference time parameter to validate

    Raises
    ------
    ValidationError
        If ref_time is invalid
    """
    if isinstance(ref_time, (int, np.integer)):
        if ref_time < 0:
            raise ValidationError("ref_time must be non-negative", "ref_time", ref_time)
    else:
        try:
            values = list(ref_time)
        except Exception:
            raise ValidationError(
                "ref_time must be an int or an iterable of ints",
                "ref_time",
                type(ref_time),
            )
        if len(values) == 0:
            raise ValidationError(
                "ref_time iterable must be non-empty", "ref_time", values
            )
        if np.any(np.array(values) < 0):
            raise ValidationError(
                "ref_time iterable values must be non-negative", "ref_time", values
            )


def _normalize_ref_time(ref_time: object, nobs: int) -> int:
    """
    Normalize ref_time to an integer suitable for slicing [:ref_time].

    - If ref_time is an int, return it clamped to [0, nobs].
    - If ref_time is an iterable (e.g., range/list/ndarray), use max(ref_time),
      then clamp to [0, nobs].
    """
    if isinstance(ref_time, (int, np.integer)):
        idx = int(ref_time)
    else:
        try:
            values = list(ref_time)
        except Exception:
            raise ValidationError(
                "ref_time must be an int or an iterable of ints",
                "ref_time",
                type(ref_time),
            )
        if len(values) == 0:
            raise ValidationError(
                "ref_time iterable must be non-empty", "ref_time", values
            )
        if np.any(np.array(values) < 0):
            raise ValidationError(
                "ref_time iterable values must be non-negative", "ref_time", values
            )
        idx = int(np.max(values))

    if idx < 0:
        raise ValidationError("ref_time must be non-negative", "ref_time", idx)
    if idx > nobs:
        idx = nobs
    return idx


def _compute_causality_at_timepoint(
    t: int,
    event_data: np.ndarray,
    stats: Dict,
    causality_measures: Dict[str, np.ndarray],
    nvar: int,
    ntrials: int,
    ref_time: int,
    estim_mode: str,
    morder: int,
    diag_flag: bool,
    old_version: bool,
) -> None:
    """
    Compute causality measures for a specific time point.

    Parameters
    ----------
    t : int
        Time point index
    event_data : np.ndarray
        Event data array
    stats : Dict
        Model statistics
    causality_measures : Dict[str, np.ndarray]
        Dictionary to store results
    nvar : int
        Number of variables
    ntrials : int
        Number of trials
    ref_time : int
        Reference time index
    estim_mode : str
        Estimation mode
    morder : int
        Model order
    diag_flag : bool
        Whether to use diagonal covariance
    old_version : bool
        Whether to use old version of rDCS
    """
    lagged_vars = event_data[2:, t, :]

    coeff = stats[estim_mode]["At"][t, :, :]
    residual_cov = stats[estim_mode]["Sigma_Et"][t, :, :]

    a_square = coeff.reshape(nvar, nvar, morder, order="F")
    b = a_square[0, 1, :]  # X -> Y coupling
    c = a_square[1, 0, :]  # Y -> X coupling

    sigy = residual_cov[0, 0] or np.finfo(float).eps
    sigx = residual_cov[1, 1] or np.finfo(float).eps

    x_past_start = 3
    y_past_start = 2
    x_past_indices = slice(x_past_start, x_past_start + 2 * morder, 2)
    y_past_indices = slice(y_past_start, y_past_start + 2 * morder, 2)

    cov_xp = stats["Sigma"][t, x_past_indices, x_past_indices]  # X past covariance
    cov_yp = stats["Sigma"][t, y_past_indices, y_past_indices]  # Y past covariance
    c_xyp = stats["Sigma"][
        t, x_past_indices, y_past_indices
    ]  # X past - Y past cross-covariance
    c_yxp = stats["Sigma"][
        t, y_past_indices, x_past_indices
    ]  # Y past - X past cross-covariance
    mean_xp = stats["mean"][x_past_indices, t]  # X past mean
    mean_yp = stats["mean"][y_past_indices, t]  # Y past mean

    cov_xp_reg = regularize_if_singular(cov_xp)
    cov_yp_reg = regularize_if_singular(cov_yp)

    _compute_transfer_entropy(
        t,
        b,
        c,
        sigy,
        sigx,
        cov_xp,
        cov_yp,
        c_xyp,
        c_yxp,
        cov_xp_reg,
        cov_yp_reg,
        causality_measures,
    )

    mean_x_ref = np.mean(
        stats["mean"][x_past_indices, :ref_time], axis=1
    )  # X past mean
    mean_y_ref = np.mean(
        stats["mean"][y_past_indices, :ref_time], axis=1
    )  # Y past mean

    cov_xp_ref = (
        cov_xp
        + mean_xp @ mean_xp.T
        - mean_xp @ mean_x_ref.T
        - mean_x_ref @ mean_xp.T
        + mean_x_ref @ mean_x_ref.T
    )
    cov_yp_ref = (
        cov_yp
        + mean_yp @ mean_yp.T
        - mean_yp @ mean_y_ref.T
        - mean_y_ref @ mean_yp.T
        + mean_y_ref @ mean_y_ref.T
    )

    ref_cov_xp = np.mean(
        stats["Sigma"][:ref_time, x_past_indices, x_past_indices], axis=0
    )  # X past
    ref_cov_yp = np.mean(
        stats["Sigma"][:ref_time, y_past_indices, y_past_indices], axis=0
    )  # Y past

    _compute_causal_strength_measures(
        t,
        b,
        c,
        sigy,
        sigx,
        cov_xp,
        cov_yp,
        cov_xp_ref,
        cov_yp_ref,
        ref_cov_xp,
        ref_cov_yp,
        lagged_vars,
        stats,
        ntrials,
        ref_time,
        diag_flag,
        old_version,
        causality_measures,
    )


def _compute_transfer_entropy(
    t: int,
    b: np.ndarray,
    c: np.ndarray,
    sigy: float,
    sigx: float,
    cov_xp: np.ndarray,
    cov_yp: np.ndarray,
    c_xyp: np.ndarray,
    c_yxp: np.ndarray,
    cov_xp_reg: np.ndarray,
    cov_yp_reg: np.ndarray,
    causality_measures: Dict[str, np.ndarray],
) -> None:
    """
    Compute Transfer Entropy for a specific time point.

    Parameters
    ----------
    t : int
        Time point index
    b : np.ndarray
        X -> Y coupling coefficients
    c : np.ndarray
        Y -> X coupling coefficients
    sigy : float
        Y residual variance
    sigx : float
        X residual variance
    cov_xp : np.ndarray
        X past covariance
    cov_yp : np.ndarray
        Y past covariance
    c_xyp : np.ndarray
        X past - Y past cross-covariance
    c_yxp : np.ndarray
        Y past - X past cross-covariance
    cov_xp_reg : np.ndarray
        Regularized X past covariance
    cov_yp_reg : np.ndarray
        Regularized Y past covariance
    causality_measures : Dict[str, np.ndarray]
        Dictionary to store results
    """
    # TE(X -> Y)
    causality_measures["TE"][t, 1] = 0.5 * np.log(
        (
            sigy
            + b.T @ cov_xp @ b
            - b.T @ c_xyp @ np.linalg.inv(cov_yp_reg) @ c_xyp.T @ b
        )
        / sigy
    )

    # TE(Y -> X)
    causality_measures["TE"][t, 0] = 0.5 * np.log(
        (
            sigx
            + c.T @ cov_yp @ c
            - c.T @ c_yxp @ np.linalg.inv(cov_xp_reg) @ c_yxp.T @ c
        )
        / sigx
    )


def _compute_causal_strength_measures(
    t: int,
    b: np.ndarray,
    c: np.ndarray,
    sigy: float,
    sigx: float,
    cov_xp: np.ndarray,
    cov_yp: np.ndarray,
    cov_xp_ref: np.ndarray,
    cov_yp_ref: np.ndarray,
    ref_cov_xp: np.ndarray,
    ref_cov_yp: np.ndarray,
    lagged_vars: np.ndarray,
    stats: Dict,
    ntrials: int,
    ref_time: int,
    diag_flag: bool,
    old_version: bool,
    causality_measures: Dict[str, np.ndarray],
) -> None:
    """
    Compute Dynamic Causal Strength (DCS) and Relative Dynamic Causal Strength (rDCS).

    Parameters
    ----------
    t : int
        Time point index
    b : np.ndarray
        X -> Y coupling coefficients
    c : np.ndarray
        Y -> X coupling coefficients
    sigy : float
        Y residual variance
    sigx : float
        X residual variance
    cov_xp : np.ndarray
        X past covariance
    cov_yp : np.ndarray
        Y past covariance
    cov_xp_ref : np.ndarray
        Reference-adjusted X past covariance
    cov_yp_ref : np.ndarray
        Reference-adjusted Y past covariance
    ref_cov_xp : np.ndarray
        Reference X past covariance
    ref_cov_yp : np.ndarray
        Reference Y past covariance
    lagged_vars : np.ndarray
        Lagged variables
    stats : Dict
        Model statistics
    ntrials : int
        Number of trials
    ref_time : int
        Reference time index
    diag_flag : bool
        Whether to use diagonal covariance
    old_version : bool
        Whether to use old version of rDCS
    causality_measures : Dict[str, np.ndarray]
        Dictionary to store results
    """
    if not diag_flag:
        causality_measures["DCS"][t, 1] = 0.5 * np.log((sigy + b.T @ cov_xp @ b) / sigy)
        causality_measures["DCS"][t, 0] = 0.5 * np.log((sigx + c.T @ cov_yp @ c) / sigx)

        if old_version:
            _compute_old_version_rdcs(
                t,
                b,
                c,
                sigy,
                sigx,
                ref_cov_xp,
                ref_cov_yp,
                lagged_vars,
                stats,
                ntrials,
                ref_time,
                causality_measures,
            )
        else:
            _compute_new_version_rdcs(
                t,
                b,
                c,
                sigy,
                sigx,
                cov_xp_ref,
                cov_yp_ref,
                ref_cov_xp,
                ref_cov_yp,
                causality_measures,
            )
    else:
        causality_measures["DCS"][t, 1] = 0.5 * np.log(
            (sigy + b.T @ np.diag(np.diag(cov_xp)) @ b) / sigy
        )
        causality_measures["DCS"][t, 0] = 0.5 * np.log(
            (sigx + c.T @ np.diag(np.diag(cov_yp)) @ c) / sigx
        )

        if old_version:
            _compute_old_version_rdcs_diagonal(
                t,
                b,
                c,
                sigy,
                sigx,
                ref_cov_xp,
                ref_cov_yp,
                lagged_vars,
                stats,
                ntrials,
                ref_time,
                causality_measures,
            )
        else:
            _compute_new_version_rdcs_diagonal(
                t,
                b,
                c,
                sigy,
                sigx,
                cov_xp_ref,
                cov_yp_ref,
                ref_cov_xp,
                ref_cov_yp,
                causality_measures,
            )


def _compute_old_version_rdcs(
    t: int,
    b: np.ndarray,
    c: np.ndarray,
    sigy: float,
    sigx: float,
    ref_cov_xp: np.ndarray,
    ref_cov_yp: np.ndarray,
    lagged_vars: np.ndarray,
    stats: Dict,
    ntrials: int,
    ref_time: int,
    causality_measures: Dict[str, np.ndarray],
) -> None:
    """Compute old version of rDCS with full covariance."""
    cov_xp_lag = (
        np.dot(
            lagged_vars - np.mean(stats["mean"][3:, :ref_time], axis=1),
            (lagged_vars - np.mean(stats["mean"][3:, :ref_time], axis=1)).T,
        )
        / ntrials
    )

    causality_measures["rDCS"][t, 1] = (
        0.5 * np.log((sigy + b.T @ ref_cov_xp @ b) / sigy)
        - 0.5
        + 0.5
        * (sigy + b.T @ cov_xp_lag[2::2, 2::2] @ b)
        / (sigy + b.T @ ref_cov_xp @ b)
    )

    causality_measures["rDCS"][t, 0] = (
        (0.5 * np.log((sigx + c.T @ ref_cov_yp @ c) / sigx))
        - 0.5
        + (
            0.5
            * (sigx + c.T @ cov_xp_lag[1::2, 1::2] @ c)
            / (sigx + c.T @ ref_cov_yp @ c)
        )
    )


def _compute_new_version_rdcs(
    t: int,
    b: np.ndarray,
    c: np.ndarray,
    sigy: float,
    sigx: float,
    cov_xp_ref: np.ndarray,
    cov_yp_ref: np.ndarray,
    ref_cov_xp: np.ndarray,
    ref_cov_yp: np.ndarray,
    causality_measures: Dict[str, np.ndarray],
) -> None:
    """Compute new version of rDCS with full covariance."""
    causality_measures["rDCS"][t, 1] = (
        (0.5 * np.log((sigy + b.T @ ref_cov_xp @ b) / sigy))
        - 0.5
        + (0.5 * (sigy + b.T @ cov_xp_ref @ b) / (sigy + b.T @ ref_cov_xp @ b))
    )

    causality_measures["rDCS"][t, 0] = (
        (0.5 * np.log((sigx + c.T @ ref_cov_yp @ c) / sigx))
        - 0.5
        + (0.5 * (sigx + c.T @ cov_yp_ref @ c) / (sigx + c.T @ ref_cov_yp @ c))
    )


def _compute_old_version_rdcs_diagonal(
    t: int,
    b: np.ndarray,
    c: np.ndarray,
    sigy: float,
    sigx: float,
    ref_cov_xp: np.ndarray,
    ref_cov_yp: np.ndarray,
    lagged_vars: np.ndarray,
    stats: Dict,
    ntrials: int,
    ref_time: int,
    causality_measures: Dict[str, np.ndarray],
) -> None:
    """Compute old version of rDCS with diagonal covariance."""
    cov_xp_lag = (
        np.dot(
            lagged_vars - np.mean(stats["mean"][3:, :ref_time], axis=1)[:, np.newaxis],
            (
                lagged_vars
                - np.mean(stats["mean"][3:, :ref_time], axis=1)[:, np.newaxis]
            ).T,
        )
        / ntrials
    )

    causality_measures["rDCS"][t, 1] = (
        (0.5 * np.log((sigy + b.T @ np.diag(np.diag(ref_cov_xp)) @ b) / sigy))
        - 0.5
        + (
            0.5
            * (sigy + b.T @ np.diag(np.diag(cov_xp_lag[2::2, 2::2])) @ b)
            / (sigy + b.T @ np.diag(np.diag(ref_cov_xp)) @ b)
        )
    )

    causality_measures["rDCS"][t, 0] = (
        (0.5 * np.log((sigx + c.T @ np.diag(np.diag(ref_cov_yp)) @ c) / sigx))
        - 0.5
        + (
            0.5
            * (sigx + c.T @ np.diag(np.diag(cov_xp_lag[1::2, 1::2])) @ c)
            / (sigx + c.T @ np.diag(np.diag(ref_cov_yp)) @ c)
        )
    )


def _compute_new_version_rdcs_diagonal(
    t: int,
    b: np.ndarray,
    c: np.ndarray,
    sigy: float,
    sigx: float,
    cov_xp_ref: np.ndarray,
    cov_yp_ref: np.ndarray,
    ref_cov_xp: np.ndarray,
    ref_cov_yp: np.ndarray,
    causality_measures: Dict[str, np.ndarray],
) -> None:
    """Compute new version of rDCS with diagonal covariance."""
    causality_measures["rDCS"][t, 1] = (
        (0.5 * np.log((sigy + b.T @ np.diag(np.diag(ref_cov_xp)) @ b) / sigy))
        - 0.5
        + (
            0.5
            * (sigy + b.T @ np.diag(np.diag(cov_xp_ref)) @ b)
            / (sigy + b.T @ np.diag(np.diag(ref_cov_xp)) @ b)
        )
    )

    causality_measures["rDCS"][t, 0] = (
        (0.5 * np.log((sigx + c.T @ np.diag(np.diag(ref_cov_yp)) @ c) / sigx))
        - 0.5
        + (
            0.5
            * (sigx + c.T @ np.diag(np.diag(cov_yp_ref)) @ c)
            / (sigx + c.T @ np.diag(np.diag(ref_cov_yp)) @ c)
        )
    )
