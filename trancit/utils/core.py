"""
Core utilities for Dynamic Causal Strength (DCS).

This module provides fundamental utilities for event extraction, statistics computation,
and DeSnap analysis. These functions form the backbone of the DCS pipeline.
"""

import logging
from typing import Dict, Union

import numpy as np

from ..config import DeSnapParams
from ..core.exceptions import ComputationError, ValidationError
from .preprocess import regularize_if_singular
from .residuals import estimate_residuals

logger = logging.getLogger(__name__)


def extract_event_windows(
    signal: np.ndarray, centers: np.ndarray, start_offset: int, window_length: int
) -> np.ndarray:
    """
    Extract windows of data from a signal around specified center points.

    This function extracts fixed-length windows from a 1D signal around each
    specified center point. The windows are extracted with a specified offset
    from the center and have a fixed length.

    Parameters
    ----------
    signal : np.ndarray
        1D array representing the signal data.
    centers : np.ndarray
        1D array of center points (indices) around which to extract windows.
    start_offset : int
        Offset from the center to start the window (can be negative).
    window_length : int
        Length of each window to extract.

    Returns
    -------
    np.ndarray
        2D array of shape (window_length, len(centers)) containing the extracted windows.
        Invalid windows are filled with NaN.

    Raises
    ------
    ValidationError
        If input parameters are invalid.
    IndexError
        If the calculated indices for any window are out of bounds for the signal array.

    Examples
    --------
    >>> import numpy as np
    >>> signal = np.random.randn(1000)
    >>> centers = np.array([100, 200, 300])
    >>> windows = extract_event_windows(signal, centers, start_offset=50, window_length=100)
    >>> print(f"Windows shape: {windows.shape}")  # (100, 3)
    """
    if not isinstance(signal, np.ndarray) or signal.ndim != 1:
        raise ValidationError(
            "signal must be a 1D numpy array",
            "signal_ndim",
            signal.ndim if hasattr(signal, "ndim") else None,
        )

    if not isinstance(centers, np.ndarray) or centers.ndim != 1:
        raise ValidationError(
            "centers must be a 1D numpy array",
            "centers_ndim",
            centers.ndim if hasattr(centers, "ndim") else None,
        )

    if not isinstance(start_offset, int):
        raise ValidationError(
            "start_offset must be an integer", "start_offset_type", type(start_offset)
        )

    if not isinstance(window_length, int) or window_length <= 0:
        raise ValidationError(
            "window_length must be a positive integer", "window_length", window_length
        )

    if len(centers) == 0:
        logger.warning("Empty centers array provided")
        return np.empty((window_length, 0))

    event_windows = np.full((window_length, len(centers)), np.nan)

    for i, center in enumerate(centers):
        try:
            start_idx = int(np.round(center - start_offset))
            end_idx = start_idx + window_length
            idx = np.arange(start_idx, end_idx)

            if np.any(idx < 0) or np.any(idx >= len(signal)):
                logger.warning(
                    f"Window {i} (center={center}) out of bounds: "
                    f"indices {idx[0]}-{idx[-1]} for signal length {len(signal)}"
                )
                continue

            event_windows[:, i] = signal[idx]

        except Exception as e:
            logger.error(f"Failed to extract window {i} (center={center}): {e}")
            continue

    valid_windows = np.sum(~np.isnan(event_windows[0, :]))
    logger.info(f"Successfully extracted {valid_windows}/{len(centers)} windows")

    return event_windows


def compute_event_statistics(
    event_data: np.ndarray, model_order: int, epsilon: float = 1e-5
) -> Dict[str, Union[np.ndarray, Dict]]:
    """
    Compute conditional statistics for VAR time series events.

    This function computes mean, covariance, and OLS coefficients for VAR time series
    events. It handles both homogeneous and inhomogeneous VAR models.

    Parameters
    ----------
    event_data : np.ndarray
        VAR time series events of shape (nvar * (model_order + 1), time_points, trials).
    model_order : int
        The model order for the VAR process.
    epsilon : float, optional
        Small value for regularization if the matrix is singular. Default is 1e-4.

    Returns
    -------
    Dict[str, Union[np.ndarray, Dict]]
        Dictionary containing the conditional statistics:
            - 'mean': Mean of the events (shape: (nvar * (model_order + 1), time_points))
            - 'Sigma': Covariance matrices
              (shape: (time_points, nvar * (model_order + 1), nvar * (model_order + 1)))
            - 'OLS': Dictionary with:
                - 'At': OLS coefficients (shape: (time_points, nvar, nvar * model_order))
                - 'bt': Residual biases
                - 'Sigma_Et': Residual covariance
                - 'sigma_Et': Residual standard deviation

    Raises
    ------
    ValidationError
        If input parameters are invalid.
    ComputationError
        If computation fails due to numerical issues.

    Examples
    --------
    >>> import numpy as np
    >>> event_data = np.random.randn(6, 50, 10)  # (nvar * (morder + 1), time, trials)
    >>> stats = compute_conditional_event_statistics(event_data, model_order=2)
    >>> print(f"Mean shape: {stats['mean'].shape}")
    >>> print(f"Sigma shape: {stats['Sigma'].shape}")
    """
    if not isinstance(event_data, np.ndarray) or event_data.ndim != 3:
        raise ValidationError(
            "event_data must be a 3D numpy array",
            "event_data_ndim",
            event_data.ndim if hasattr(event_data, "ndim") else None,
        )

    if not isinstance(model_order, int) or model_order <= 0:
        raise ValidationError(
            "model_order must be a positive integer", "model_order", model_order
        )

    if not isinstance(epsilon, (int, float)) or epsilon <= 0:
        raise ValidationError("epsilon must be a positive number", "epsilon", epsilon)

    nvar = event_data.shape[0] // (model_order + 1)
    if event_data.shape[0] != nvar * (model_order + 1):
        raise ValidationError(
            f"event_data shape {event_data.shape[0]} is not compatible "
            f"with model_order {model_order}",
            "event_data_shape",
            event_data.shape,
        )

    if event_data.shape[2] < 2:
        raise ValidationError(
            "At least 2 trials required for statistics computation",
            "n_trials",
            event_data.shape[2],
        )

    try:
        stats = {
            "mean": np.mean(event_data, axis=2),
            "Sigma": np.zeros(
                (
                    event_data.shape[1],
                    nvar * (model_order + 1),
                    nvar * (model_order + 1),
                )
            ),
            "OLS": {"At": np.zeros((event_data.shape[1], nvar, nvar * model_order))},
        }

        for t in range(event_data.shape[1]):
            try:
                temp = event_data[:, t, :] - stats["mean"][:, t, np.newaxis]
                stats["Sigma"][t, :, :] = np.dot(temp, temp.T) / event_data.shape[2]

                Sigma_sub_matrix = stats["Sigma"][t, :nvar, nvar:]
                Sigma_past = stats["Sigma"][t, nvar:, nvar:]

                if np.linalg.det(Sigma_past) > epsilon:
                    Sigma_past_inv = np.linalg.inv(Sigma_past)
                    stats["OLS"]["At"][t, :, :] = np.dot(
                        Sigma_sub_matrix, Sigma_past_inv
                    )
                else:
                    logger.warning(
                        f"Singular Sigma_past at time {t}, using regularization"
                    )
                    Sigma_past_regularized = regularize_if_singular(
                        Sigma_past, epsilon=epsilon
                    )
                    stats["OLS"]["At"][t, :, :] = np.linalg.solve(
                        Sigma_past_regularized, Sigma_sub_matrix.T
                    ).T

            except Exception as e:
                logger.error(f"Failed to compute statistics at time {t}: {e}")
                raise ComputationError(
                    f"Statistics computation failed at time {t}",
                    "statistics_computation",
                    (t,),
                )

        stats["OLS"]["bt"], stats["OLS"]["Sigma_Et"], stats["OLS"]["sigma_Et"] = (
            estimate_residuals(stats)
        )

        logger.info(
            f"Successfully computed statistics for {event_data.shape[1]} time points"
        )
        return stats

    except Exception as e:
        logger.error(f"Statistics computation failed: {e}")
        raise ComputationError(
            f"Statistics computation failed: {e}", "statistics_computation", ()
        )


def extract_event_snapshots(
    time_series: np.ndarray,
    locations: np.ndarray,
    model_order: int,
    lag_step: int,
    start_offset: int,
    extract_length: int,
) -> np.ndarray:
    """
    Extract event snapshots from time series data.

    This function extracts fixed-length snapshots from multivariate time series
    data around specified event locations. Each snapshot includes the current
    time point and lagged data according to the model order.

    Parameters
    ----------
    time_series : np.ndarray
        Time series data of shape (n_vars, n_time_points).
    locations : np.ndarray
        1D array of event location indices.
    model_order : int
        Model order (number of lags to include).
    lag_step : int
        Step size between lags.
    start_offset : int
        Offset from event location to start extraction.
    extract_length : int
        Length of each snapshot to extract.

    Returns
    -------
    np.ndarray
        3D array of shape (n_vars * (model_order + 1), extract_length, n_events)
        containing the extracted snapshots.

    Raises
    ------
    ValidationError
        If input parameters are invalid.
    IndexError
        If any event location is out of bounds.

    Examples
    --------
    >>> import numpy as np
    >>> time_series = np.random.randn(2, 1000)  # (n_vars, time)
    >>> locations = np.array([100, 200, 300])
    >>> snapshots = extract_event_snapshots(
    ...     time_series, locations, model_order=4, lag_step=1,
    ...     start_offset=50, extract_length=100
    ... )
    >>> print(f"Snapshots shape: {snapshots.shape}")  # (10, 100, 3)
    """
    if not isinstance(time_series, np.ndarray) or time_series.ndim != 2:
        raise ValidationError(
            "time_series must be a 2D numpy array",
            "time_series_ndim",
            time_series.ndim if hasattr(time_series, "ndim") else None,
        )

    if not isinstance(locations, np.ndarray) or locations.ndim != 1:
        raise ValidationError(
            "locations must be a 1D numpy array",
            "locations_ndim",
            locations.ndim if hasattr(locations, "ndim") else None,
        )

    if not isinstance(model_order, int) or model_order <= 0:
        raise ValidationError(
            "model_order must be a positive integer", "model_order", model_order
        )

    if not isinstance(lag_step, int) or lag_step <= 0:
        raise ValidationError(
            "lag_step must be a positive integer", "lag_step", lag_step
        )

    if not isinstance(start_offset, int):
        raise ValidationError(
            "start_offset must be an integer", "start_offset", start_offset
        )

    if not isinstance(extract_length, int) or extract_length <= 0:
        raise ValidationError(
            "extract_length must be a positive integer",
            "extract_length",
            extract_length,
        )

    n_vars, n_time_points = time_series.shape

    if len(locations) == 0:
        logger.warning("Empty locations array provided")
        return np.empty((n_vars * (model_order + 1), extract_length, 0))

    min_location = np.min(locations)
    max_location = np.max(locations)

    if min_location < 0 or max_location >= n_time_points:
        raise IndexError(
            f"Event locations {min_location}-{max_location} out of bounds "
            f"for time series length {n_time_points}"
        )

    try:
        n_events = len(locations)
        snapshots = np.zeros((n_vars * (model_order + 1), extract_length, n_events))

        idx1 = np.arange(n_vars * (model_order + 1))
        idx2 = np.tile(np.arange(n_vars), model_order + 1)
        delay = np.tile(np.arange(0, model_order + 1) * lag_step, (n_vars, 1)).flatten()

        for n in range(len(idx1)):
            snapshots[idx1[n], :, :] = extract_event_windows(
                time_series[idx2[n], :],
                locations - delay[n],
                start_offset,
                extract_length,
            )
        valid_snapshots = np.sum(~np.isnan(snapshots[0, 0, :]))
        logger.info(f"Successfully extracted {valid_snapshots}/{n_events} snapshots")

        return snapshots

    except Exception as e:
        logger.error(f"Snapshot extraction failed: {e}")
        raise ComputationError(
            f"Snapshot extraction failed: {e}", "snapshot_extraction", None
        )


def perform_desnap_analysis(inputs: DeSnapParams) -> Dict[str, Union[np.ndarray, Dict]]:
    """
    Perform DeSnap analysis for event-based causality.

    This function implements the DeSnap (De-Snapshot) analysis method for
    event-based causality analysis. It performs a "de-snapshotting" analysis
    to derive unconditional statistics from conditional statistics by accounting
    for a conditioning variable 'D'.

    Parameters
    ----------
    inputs : DeSnapParams
        Configuration object containing all parameters for DeSnap analysis.

    Returns
    -------
    Dict[str, Union[np.ndarray, Dict]]
        Dictionary containing DeSnap analysis results:
            - 'loc_size': Size of locations per bin of D
            - 'p_t', 'q_t': Coefficients from the first linear regression
            - 'd_bin_bar': Mean D values for each bin
            - 'mean_Yt_cond': Mean Yt events per bin of D
            - 'mu_D': Estimated unconditional mean of D
            - 'event_stats_uncond': Unconditional statistics
            - 'cov_pt': Covariance related to p_t
            - 'c': Covariance adjustment factor

    Raises
    ------
    ValidationError
        If input parameters are invalid.
    ComputationError
        If computation fails due to numerical issues.

    Examples
    --------
    >>> from trancit.config import DeSnapParams
    >>> params = DeSnapParams(...)  # Configure parameters
    >>> results = perform_desnap_analysis(params)
    >>> print(f"Results keys: {list(results.keys())}")
    """
    if not isinstance(inputs, DeSnapParams):
        raise ValidationError(
            "inputs must be a DeSnapParams object", "inputs_type", type(inputs)
        )

    try:
        if inputs.d0_max is None and inputs.maxStdRatio is None:
            raise ValidationError(
                "Either d0_max or maxStdRatio must be provided", "d0_max_missing", None
            )

        if inputs.N_d <= 0:
            raise ValidationError(
                "N_d (number of bins) must be positive", "N_d", inputs.N_d
            )

        if inputs.detection_signal.ndim != 1:
            raise ValidationError(
                "detection_signal must be 1D",
                "detection_signal_ndim",
                inputs.detection_signal.ndim,
            )

        if inputs.original_signal.ndim != 2:
            raise ValidationError(
                "original_signal must be 2D",
                "original_signal_ndim",
                inputs.original_signal.ndim,
            )

        if inputs.detection_signal.shape[0] != inputs.original_signal.shape[1]:
            raise ValidationError(
                "detection_signal and original_signal must have same time dimension",
                "signal_lengths",
                (inputs.detection_signal.shape[0], inputs.original_signal.shape[1]),
            )

        if inputs.d0_max is None:
            if inputs.maxStdRatio is not None:
                inputs.d0_max = np.mean(
                    inputs.detection_signal
                ) + inputs.maxStdRatio * np.std(inputs.detection_signal)
            else:
                inputs.d0_max = np.max(inputs.detection_signal)

        logger.info(
            f"DeSnap analysis: d0={inputs.d0}, d0_max={inputs.d0_max}, N_d={inputs.N_d}"
        )

        bin_step = abs(inputs.d0_max - inputs.d0) / inputs.N_d
        d_bin_edges = np.arange(inputs.d0, inputs.d0_max + bin_step + 1e-12, bin_step)
        num_bins = len(d_bin_edges)

        d_bin_mean_detection = np.full(num_bins, np.nan)
        num_input_channels = inputs.original_signal.shape[0]
        num_snapshot_vars = num_input_channels * (inputs.morder + 1)
        mean_events_cond_binned = np.full(
            (num_bins, num_snapshot_vars, inputs.l_extract), np.nan
        )

        DeSnap_results = {
            "loc_size": np.full(num_bins, np.nan),
            "event_stats": inputs.event_stats,
        }

        logger.info("Processing bins of conditioning variable D ...")
        current_bin_uplim = np.max(inputs.detection_signal)

        for n, current_bin_lolim in enumerate(d_bin_edges):
            mask = (inputs.detection_signal >= current_bin_lolim) & (
                inputs.detection_signal < current_bin_uplim
            )

            d_bin_mean_detection[n] = np.mean(inputs.detection_signal[mask])
            temp_loc = np.where(mask)[0]

            valid_locs = temp_loc.copy()
            # valid_locs = valid_locs[
            #     inputs.original_signal.shape[0] - valid_locs >= inputs.l_extract - inputs.l_start
            # ]
            # valid_locs = valid_locs[
            #     valid_locs - inputs.l_start - (inputs.morder * inputs.tau) >= 0
            # ]

            DeSnap_results["loc_size"][n] = len(valid_locs)

            if len(valid_locs) > 0:
                try:
                    events_binned = extract_event_snapshots(
                        inputs.original_signal,
                        valid_locs,
                        inputs.morder,
                        inputs.tau,
                        inputs.l_start,
                        inputs.l_extract,
                    )
                    if events_binned.shape[2] > 0:
                        mean_events_cond_binned[n, :, :] = np.mean(
                            events_binned, axis=2
                        )
                    else:
                        logger.warning(
                            f"No snapshots extracted for bin {n+1} despite "
                            f"{len(valid_locs)} valid_locs"
                        )
                except Exception as e:
                    logger.error(f"Failed to extract snapshots for bin {n}: {e}")
            else:
                logger.warning(
                    f"No valid locations for snapshot extraction in bin {n+1}"
                )

        # First Linear Regression: Fit p_t and q_t
        logger.info("Performing first linear regression for p_t and q_t...")
        try:
            from .helpers import compute_multi_variable_linear_regression

            p_t, q_t = compute_multi_variable_linear_regression(
                d_bin_mean_detection, mean_events_cond_binned
            )
            DeSnap_results["p_t"] = p_t
            DeSnap_results["q_t"] = q_t
            DeSnap_results["d_bin_bar"] = d_bin_mean_detection
            DeSnap_results["mean_Yt_cond"] = mean_events_cond_binned
        except Exception as e:
            logger.error(f"First linear regression failed: {e}")
            raise ComputationError(
                f"First linear regression failed: {e}", "linear_regression_1", None
            )

        # Second Linear Regression: Estimate mu_D (unconditional mean of D)
        p_t_flat = p_t.reshape(-1, 1)
        q_t_flat = -q_t.reshape(-1)

        nan_mask_regression2 = ~np.isnan(p_t_flat.ravel()) & ~np.isnan(q_t_flat)
        if not np.any(nan_mask_regression2):
            raise ComputationError(
                "All p_t or q_t values are NaN, cannot compute mu_D",
                "mu_D_computation",
                None,
            )

        p_t_flat_valid = p_t_flat[nan_mask_regression2]
        q_t_flat_valid = q_t_flat[nan_mask_regression2]

        try:
            DeSnap_results["mu_D"] = np.linalg.lstsq(
                p_t_flat_valid, q_t_flat_valid, rcond=None
            )[0][0]
        except Exception as e:
            logger.error(f"Second linear regression failed: {e}")
            raise ComputationError(
                f"Second linear regression failed: {e}", "linear_regression_2", None
            )

        DeSnap_results["event_stats_uncond"] = {}
        DeSnap_results["event_stats_uncond"]["mean"] = (
            q_t + p_t * DeSnap_results["mu_D"]
        )

        # Third Linear Regression: Compute Covariance Adjustment Factor 'c'
        logger.info(
            "Performing third linear regression for covariance adjustment factor 'c'..."
        )
        DeSnap_results["cov_pt"] = np.full(
            (inputs.l_extract, num_snapshot_vars, num_snapshot_vars), np.nan
        )
        for t in range(inputs.l_extract):
            DeSnap_results["cov_pt"][t, :, :] = np.outer(p_t[:, t], p_t[:, t])

        try:
            if inputs.diff_flag:
                x_reg_c = np.diff(DeSnap_results["cov_pt"], axis=0)
                y_reg_c = np.diff(inputs.event_stats["Sigma"], axis=0)
                DeSnap_results["c"] = np.linalg.lstsq(
                    x_reg_c.reshape(-1, 1), y_reg_c.reshape(-1), rcond=None
                )[0][0]
            else:
                x_reg_c_levels = DeSnap_results["cov_pt"][:, 0, 0]
                y_reg_c_levels = inputs.event_stats["Sigma"][:, 0, 0]
                X_design_c = np.vstack([np.ones_like(x_reg_c_levels), x_reg_c_levels]).T
                temp_coeffs_c = np.linalg.lstsq(X_design_c, y_reg_c_levels, rcond=None)[
                    0
                ]
                DeSnap_results["c"] = temp_coeffs_c[1]
        except Exception as e:
            logger.error(f"Third linear regression failed: {e}")
            raise ComputationError(
                f"Third linear regression failed: {e}", "linear_regression_3", None
            )

        DeSnap_results["event_stats_uncond"]["Sigma"] = (
            inputs.event_stats["Sigma"] - DeSnap_results["c"] * DeSnap_results["cov_pt"]
        )

        logger.info("Calculating unconditional AR coefficients...")
        try:
            nvar_actual = inputs.event_stats["OLS"]["At"].shape[1]
        except (KeyError, AttributeError, IndexError):
            raise ValidationError(
                "Could not determine 'nvar_actual' from inputs.event_stats['OLS']['At']",
                "ols_at_structure",
                None,
            )

        DeSnap_results["event_stats_uncond"]["OLS"] = {}
        DeSnap_results["event_stats_uncond"]["OLS"]["At"] = np.full(
            (inputs.l_extract, nvar_actual, nvar_actual * inputs.morder), np.nan
        )

        for t in range(inputs.l_extract):
            try:
                Sigma_yx_uncond = DeSnap_results["event_stats_uncond"]["Sigma"][
                    t, :nvar_actual, nvar_actual:
                ]
                Sigma_xx_uncond = DeSnap_results["event_stats_uncond"]["Sigma"][
                    t, nvar_actual:, nvar_actual:
                ]

                Sigma_xx_uncond_reg = regularize_if_singular(Sigma_xx_uncond)
                if not np.allclose(Sigma_xx_uncond, Sigma_xx_uncond_reg):
                    logger.warning(
                        f"DeSnap: Applied regularization to Sigma_xx_uncond at time step {t}"
                    )

                DeSnap_results["event_stats_uncond"]["OLS"]["At"][t, :, :] = (
                    Sigma_yx_uncond @ np.linalg.inv(Sigma_xx_uncond_reg)
                )
            except np.linalg.LinAlgError:
                logger.warning(
                    f"DeSnap: Singular matrix at time step {t}, using pseudo-inverse"
                )
                DeSnap_results["event_stats_uncond"]["OLS"]["At"][t, :, :] = (
                    Sigma_yx_uncond @ np.linalg.pinv(Sigma_xx_uncond_reg)
                )
            except Exception as e:
                logger.error(f"Failed to compute AR coefficients at time step {t}: {e}")
                continue

        logger.info(f"Successfully completed DeSnap analysis with {num_bins} bins")
        return DeSnap_results

    except Exception as e:
        logger.error(f"DeSnap analysis failed: {e}")
        raise ComputationError(f"DeSnap analysis failed: {e}", "desnap_analysis", ())
