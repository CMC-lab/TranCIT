import logging
from typing import Dict

import numpy as np

from ..config import DeSnapParams
from .helpers import compute_multi_variable_linear_regression
from .preprocess import regularize_if_singular
from .residuals import estimate_residuals

logging.basicConfig(level=logging.INFO)


def extract_event_windows(
    signal: np.ndarray, centers: np.ndarray, start_offset: int, window_length: int
) -> np.ndarray:
    """
    Extract windows of data from a signal around specified center points.

    Parameters
    ----------
    signal : np.ndarray
        1D array representing the signal data.
    centers : np.ndarray
        1D array of center points (indices) around which to extract windows.
    start_offset : int
        Offset from the center to start the window.
    window_length : int
        Length of each window to extract.

    Returns
    -------
    np.ndarray
        2D array of shape (window_length, len(centers)) containing the extracted windows.

    Raises
    ------
    IndexError
        If the calculated indices for any window are out of bounds for the signal array.
    """
    event_windows = np.full((window_length, len(centers)), np.nan)

    for i, center in enumerate(centers):
        start_idx = int(np.round(center - start_offset))
        end_idx = start_idx + window_length
        idx = np.arange(start_idx, end_idx)

        if np.any(idx < 0) or np.any(idx >= len(signal)):
            logging.error(
                f"Index out of bounds for center {center}: {idx} for signal of length {len(signal)}"
            )
            raise IndexError(
                f"Index out of bounds: {idx} for array of length {len(signal)}"
            )

        event_windows[:, i] = signal[idx]

    return event_windows


def compute_conditional_event_statistics(
    event_data: np.ndarray, model_order: int, epsilon: float = 1e-4
) -> Dict:
    """
    Compute conditional statistics for VAR time series events, including mean and covariance.

    Parameters
    ----------
    event_data : np.ndarray
        VAR time series events of shape (nvar * (model_order + 1), time points, trials).
    model_order : int
        The model order for the VAR process.
    epsilon : float, optional
        Small value for regularization if the matrix is singular. Default is 1e-4.

    Returns
    -------
    dict
        Dictionary containing the conditional statistics:
            - 'mean': Mean of the events (shape: (nvar * (model_order + 1), time points)).
            - 'Sigma': Covariance matrices (shape: (time points, nvar * (model_order + 1), nvar * (model_order + 1))).
            - 'OLS': Dictionary with:
                - 'At': OLS coefficients (shape: (time points, nvar, nvar * model_order)).
                - 'bt': Residual biases.
                - 'Sigma_Et': Residual covariance.
                - 'sigma_Et': Residual standard deviation.
    """
    nvar = event_data.shape[0] // (model_order + 1)
    stats = {
        "mean": np.mean(event_data, axis=2),
        "Sigma": np.zeros(
            (event_data.shape[1], nvar * (model_order + 1), nvar * (model_order + 1))
        ),
        "OLS": {"At": np.zeros((event_data.shape[1], nvar, nvar * model_order))},
    }

    for t in range(event_data.shape[1]):
        temp = event_data[:, t, :] - stats["mean"][:, t, np.newaxis]
        stats["Sigma"][t, :, :] = np.dot(temp, temp.T) / event_data.shape[2]

        Sigma_sub_matrix = stats["Sigma"][t, :nvar, nvar:]
        Sigma_end = stats["Sigma"][t, nvar:, nvar:]

        if np.linalg.det(Sigma_end) == 0:
            logging.warning(
                f"Matrix singular at time {t}, applying regularization with epsilon={epsilon}"
            )
            Sigma_end = regularize_if_singular(Sigma_end, epsilon)

        stats["OLS"]["At"][t, :, :] = np.dot(Sigma_sub_matrix, np.linalg.inv(Sigma_end))

    stats["OLS"]["bt"], stats["OLS"]["Sigma_Et"], stats["OLS"]["sigma_Et"] = (
        estimate_residuals(stats)
    )
    return stats


def extract_event_snapshots(
    time_series: np.ndarray,
    locations: np.ndarray,
    model_order: int,
    lag_step: int,
    start_offset: int,
    extract_length: int,
) -> np.ndarray:
    """
    Extract event snapshots from time series data for multiple variables and lags.

    Parameters
    ----------
    time_series : np.ndarray
        2D array of shape (variables, time points) containing the time series data.
    locations : np.ndarray
        1D array of event locations (indices).
    model_order : int
        The model order (number of lags).
    lag_step : int
        The step size for lags.
    start_offset : int
        Offset from the location to start the window.
    extract_length : int
        Length of each extracted window.

    Returns
    -------
    np.ndarray
        3D array of shape (variables * (model_order + 1), extract_length, len(locations))
        containing the extracted event snapshots.
    """
    nvar = time_series.shape[0]
    snapshots = np.full(
        (nvar * (model_order + 1), extract_length, len(locations)), np.nan
    )

    idx1 = np.arange(nvar * (model_order + 1))
    idx2 = np.tile(np.arange(nvar), model_order + 1)
    delay = np.tile(np.arange(0, model_order + 1) * lag_step, (nvar, 1)).flatten()

    for n in range(len(idx1)):
        snapshots[idx1[n], :, :] = extract_event_windows(
            time_series[idx2[n], :], locations - delay[n], start_offset, extract_length
        )

    return snapshots


def compute_event_statistics(event_data: np.ndarray, model_order: int) -> Dict:
    """
    Compute statistics for VAR time series events, including mean, covariance, and OLS coefficients.

    Parameters
    ----------
    event_data : np.ndarray
        VAR time series events of shape (variables * (model_order + 1), time points, trials).
    model_order : int
        The model order for the VAR process.

    Returns
    -------
    dict
        Dictionary containing the event statistics:
            - 'mean': Mean of the events (shape: (variables * (model_order + 1), time points)).
            - 'n_trials': Number of trials.
            - 'Sigma': Covariance matrices (shape: (time points, nvar * (model_order + 1), nvar * (model_order + 1))).
            - 'OLS': Dictionary with:
                - 'At': OLS coefficients (shape: (time points, nvar, nvar * model_order)).
                - 'bt': Residual biases.
                - 'Sigma_Et': Residual covariance.
                - 'sigma_Et': Residual standard deviation.
    """
    nvar = event_data.shape[0] // (model_order + 1)
    nobs = event_data.shape[1]
    n_trials = event_data.shape[2]

    stats = {
        "mean": np.mean(event_data, axis=2),
        "n_trials": n_trials,
        "Sigma": np.zeros((nobs, nvar * (model_order + 1), nvar * (model_order + 1))),
        "OLS": {"At": np.zeros((nobs, nvar, nvar * model_order))},
    }

    for t in range(nobs):
        temp = event_data[:, t, :] - stats["mean"][:, t][:, np.newaxis]
        stats["Sigma"][t, :, :] = np.dot(temp, temp.T) / n_trials

        Sigma_12 = stats["Sigma"][t, :nvar, nvar:]  # Shape: (nvar, nvar * model_order)
        Sigma_22 = stats["Sigma"][
            t, nvar:, nvar:
        ]  # Shape: (nvar * model_order, nvar * model_order)

        Sigma_22_reg = regularize_if_singular(Sigma_22)
        if not np.allclose(Sigma_22, Sigma_22_reg):
            logging.warning(f"Applied regularization to Sigma_22 at time step {t}")

        stats["OLS"]["At"][t, :, :] = Sigma_12 @ np.linalg.inv(Sigma_22_reg)

    stats["OLS"]["bt"], stats["OLS"]["Sigma_Et"], stats["OLS"]["sigma_Et"] = (
        estimate_residuals(stats)
    )
    return stats

def desnapanalysis(
    inputs: DeSnapParams,
) -> Dict:
    """
    Performs a "de-snapshotting" analysis to derive unconditional statistics
    from conditional statistics by accounting for a conditioning variable 'D'.

    This process involves several linear regression steps to estimate and remove
    the influence of 'D' from the mean and covariance of event-related data.

    Parameters
    ----------
    inputs : DeSnapInputs
        A dataclass object containing all necessary input parameters and data:
        - detection_signal (np.ndarray): The conditioning variable values.
        - original_signal (np.ndarray): The time series data from which snapshots are extracted.
                          Shape should be compatible with `extract_event_snapshots`.
                          Typically (n_channels, total_time_points).
        - Yt_stats_cond (Dict): Conditional statistics, usually the 'Yt_stats'
                                output from `snapshot_detect_analysis_pipeline`.
                                Expected to contain fields like 'Sigma', 'OLS.At'.
        - morder (int): Model order used for snapshot extraction.
        - tau (int): Lag step for snapshot extraction.
        - L_start (int): Start offset for snapshot extraction.
        - L_extract (int): Length of extracted snapshots.
        - d0 (float): Starting value for binning 'D'.
        - N_d (int): Number of bins for 'D'.
        - d0_max (Optional[float]): Maximum value for binning 'D'.
                                    Either d0_max or maxStdRatio must be provided.
        - maxStdRatio (Optional[float]): Alternative to d0_max, defines d0_max
                                         relative to mean and std of 'D'.
        - diff_flag (bool): Flag to control method for calculating covariance
                            adjustment factor 'c'.

    Returns
    -------
    Dict[str, Any]
        A dictionary containing the results of the de-snapshotting analysis:
        - 'loc_size': Size of locations per bin of D.
        - 'p_t', 'q_t': Coefficients from the first linear regression.
        - 'd_bin_bar': Mean D values for each bin.
        - 'mean_Yt_cond': Mean Yt events per bin of D.
        - 'mu_D': Estimated unconditional mean of D.
        - 'Yt_stats_uncond' (Dict): Unconditional statistics:
            - 'mean': Unconditional mean of Yt.
            - 'Sigma': Unconditional covariance of Yt.
            - 'OLS': {'At': Unconditional AR coefficients}.
        - 'cov_pt': Covariance related to p_t.
        - 'c': Covariance adjustment factor.

    Raises
    ------
    ValueError
        If required input fields are missing or improperly specified.
    """
    if inputs.d0_max is None and inputs.maxStdRatio is None:
        raise ValueError("Either d0_max or maxStdRatio must be provided in DeSnapInputs")
    
    if inputs.d0_max is None and inputs.maxStdRatio is not None:
        d0_max_resolved = np.mean(inputs.detection_signal) + inputs.maxStdRatio * np.std(inputs.detection_signal)
    
    if inputs.N_d <= 0:
        raise ValueError("N_d (number of bins) must be positive.")
    
    if d0_max_resolved <= inputs.d0:
        raise ValueError(f"d0_max_resolved ({d0_max_resolved}) must be greater than d0 ({inputs.d0}).")
    
    bin_step = (d0_max_resolved - inputs.d0) / inputs.N_d
    d_bin_edges = np.arange(inputs.d0, d0_max_resolved + bin_step + 1e-12, bin_step)
    d_bin_lower_limits = d_bin_edges[:-1]
    
    # --- Initialize Result Arrays ---
    num_bins = len(d_bin_lower_limits)
    d_bin_mean_D = np.full(num_bins, np.nan) # Stores mean of D values in each bin
    
    num_input_channels = inputs.x.shape[0]
    num_snapshot_vars = num_input_channels * (inputs.morder + 1)
    mean_events_cond_binned = np.full((num_bins, num_snapshot_vars, inputs.l_extract), np.nan)
    
    DeSnap_results = {}
    DeSnap_results['loc_size'] = np.full(num_bins, np.nan)
    
    # --- Loop Over Bins of D to Gather Samples ---
    print("Processing bins of conditioning variable D...")
    current_bin_uplim = np.max(inputs.detection_signal)
    for n, current_bin_lolim in enumerate(d_bin_lower_limits):
        # current_bin_uplim = d_bin_edges[n + 1]
        
        mask = (inputs.detection_signal >= current_bin_lolim) & (inputs.detection_signal < current_bin_uplim)
        
        if not np.any(mask):
            print(f"Bin {n+1} ({current_bin_lolim:.2f}-{current_bin_uplim:.2f}): No D values. Skipping.")
            # d_bin_mean_D[n] remains NaN
            # mean_Yt_cond_binned[n, :, :] remains NaN
            DeSnap_results['loc_size'][n] = 0
            continue
        
        d_bin_mean_D[n] = np.mean(inputs.detection_signal[mask])
        temp_loc = np.where(mask)[0]
        
        valid_locs = temp_loc
        valid_locs = valid_locs[inputs.original_signal.shape[0] - valid_locs >= inputs.l_extract - inputs.l_start]
        valid_locs = valid_locs[valid_locs - inputs.l_start - (inputs.morder * inputs.tau) >= 0]
        
        print(f"Bin {n+1} ({current_bin_lolim:.2f}-{current_bin_uplim:.2f}): "
                f"Mean D={d_bin_mean_D[n]:.2f}, N_in_bin={len(temp_loc)}, N_valid_locs={len(valid_locs)}")
        
        DeSnap_results['loc_size'][n] = len(valid_locs)
        
        if len(valid_locs) > 0:
            events_binned = extract_event_snapshots(
                inputs.original_signal, valid_locs, inputs.morder, inputs.tau,
                inputs.l_start, inputs.l_extract
            )
            if events_binned.shape[2] > 0:
                mean_events_cond_binned[n, :, :] = np.mean(events_binned, axis=2)
            else:
                print(f"  Warning: No snapshots extracted for bin {n+1} despite {len(valid_locs)} valid_locs.")
        else:
            print(f"  No valid locations for snapshot extraction in bin {n+1}.")
    
    # Filter out bins where no data was found (d_bin_mean_D is NaN)
    valid_bins_mask = ~np.isnan(d_bin_mean_D)
    if not np.any(valid_bins_mask):
        raise ValueError("No valid data points found across all bins of D. Check binning parameters or input D.")
        
    d_bin_mean_D_filtered = d_bin_mean_D[valid_bins_mask]
    mean_events_cond_binned_filtered = mean_events_cond_binned[valid_bins_mask, :, :]
    
    # --- First Linear Regression: Fit p_t and q_t ---
    # Regress mean_Yt_cond_binned_filtered on d_bin_mean_D_filtered
    # p_t will be slopes, q_t will be intercepts
    print("Performing first linear regression for p_t and q_t...")
    p_t, q_t = compute_multi_variable_linear_regression(d_bin_mean_D_filtered, mean_events_cond_binned_filtered)
    DeSnap_results["p_t"] = p_t
    DeSnap_results["q_t"] = q_t
    DeSnap_results["d_bin_bar"] = d_bin_mean_D_filtered
    DeSnap_results["mean_Yt_cond"] = mean_events_cond_binned_filtered
    
    # --- Second Linear Regression: Estimate mu_D (unconditional mean of D) ---
    # Regress -q_t on p_t. Reshape p_t and q_t to be 1D vectors for this regression.
    # This assumes a single scalar mu_D.
    p_t_flat = p_t.reshape(-1, 1) # All p_t values as a column vector
    q_t_flat = -q_t.reshape(-1)   # All -q_t values as a 1D array
    
    # Filter out NaNs that might have resulted from p_t/q_t calculation if some bins had too few points
    # for compute_multi_variable_linear_regression (though it has its own checks)
    nan_mask_regression2 = ~np.isnan(p_t_flat.ravel()) & ~np.isnan(q_t_flat)
    if not np.any(nan_mask_regression2):
        raise ValueError("All p_t or q_t values are NaN, cannot compute mu_D.")

    p_t_flat_valid = p_t_flat[nan_mask_regression2]
    q_t_flat_valid = q_t_flat[nan_mask_regression2]
    
    # lstsq solves p_t_flat_valid * mu_D = q_t_flat_valid
    DeSnap_results['mu_D'] = np.linalg.lstsq(p_t_flat_valid, q_t_flat_valid, rcond=None)[0][0]
    # --- Compute Unconditional Mean of Yt ---
    DeSnap_results['Yt_stats_uncond'] = {}
    DeSnap_results['Yt_stats_uncond']['mean'] = q_t + p_t * DeSnap_results['mu_D']
    
    # --- Third Linear Regression: Compute Covariance Adjustment Factor 'c' ---
    # This part adjusts the conditional covariance matrices.
    # cov_pt is related to the variance explained by p_t.
    print("Performing third linear regression for covariance adjustment factor 'c'...")
    DeSnap_results['cov_pt'] = np.full((inputs.l_extract, num_snapshot_vars, num_snapshot_vars), np.nan)
    for t in range(inputs.l_extract):
        DeSnap_results['cov_pt'][t, :, :] = np.outer(p_t[:, t], p_t[:, t])
    
    if inputs.diff_flag:
        x_reg_c = np.diff(DeSnap_results['cov_pt'], axis=0) # Independent variable
        y_reg_c = np.diff(inputs.Yt_stats_cond['Sigma'], axis=0) # Dependent variable
        DeSnap_results['c'] = np.linalg.lstsq(x_reg_c.reshape(-1, 1), y_reg_c.reshape(-1), rcond=None)[0][0]
    else:
        x_reg_c_levels = DeSnap_results['cov_pt'][:, 0, 0]
        y_reg_c_levels = inputs.Yt_stats_cond['Sigma'][:, 0, 0]
        
        valid_c_mask = ~np.isnan(x_reg_c_levels) & ~np.isnan(y_reg_c_levels)
        if not np.any(valid_c_mask):
            raise ValueError("Not enough valid data points to calculate 'c' for covariance adjustment.")

        X_design_c = np.vstack([np.ones(np.sum(valid_c_mask)), x_reg_c_levels[valid_c_mask]]).T
        temp_coeffs_c = np.linalg.lstsq(X_design_c, y_reg_c_levels[valid_c_mask], rcond=None)[0]
        DeSnap_results['c'] = temp_coeffs_c[1]
    
    # --- Compute Unconditional Covariance Sigma ---
    DeSnap_results['Yt_stats_uncond']['Sigma'] = inputs.Yt_stats_cond['Sigma'] - DeSnap_results['c'] * DeSnap_results['cov_pt']
    
    # --- Compute Unconditional Autoregressive Coefficients At ---
    print("Calculating unconditional AR coefficients...")
    try:
        nvar_actual = inputs.Yt_stats_cond['OLS']['At'].shape[1]
    except (KeyError, AttributeError, IndexError):
        raise ValueError("Could not determine 'nvar_actual' from inputs.Yt_stats_cond['OLS']['At']. Ensure it's correctly structured.")
        
    DeSnap_results['Yt_stats_uncond']['OLS'] = {}
    DeSnap_results['Yt_stats_uncond']['OLS']['At'] = np.full(
        (inputs.l_extract, nvar_actual, nvar_actual * inputs.morder), np.nan
    )
    
    for t in range(inputs.l_extract):
        # Sigma_uncond is (L_extract, num_snapshot_vars, num_snapshot_vars)
        # num_snapshot_vars = nvar_actual * (morder + 1)
        # Current data part: indices 0 to nvar_actual-1
        # Lagged data part: indices nvar_actual to num_snapshot_vars-1
        Sigma_yx_uncond = DeSnap_results['Yt_stats_uncond']['Sigma'][t, :nvar_actual, nvar_actual:]
        Sigma_xx_uncond = DeSnap_results['Yt_stats_uncond']['Sigma'][t, nvar_actual:, nvar_actual:]
        
        # Solve At * Sigma_xx_uncond = Sigma_yx_uncond  => At = Sigma_yx_uncond * inv(Sigma_xx_uncond)
        try:
            DeSnap_results['Yt_stats_uncond']['OLS']['At'][t, :, :] = np.linalg.solve(Sigma_xx_uncond.T, Sigma_yx_uncond.T).T
        except np.linalg.LinAlgError:
            print(f"Warning: Singular matrix encountered for Sigma_xx_uncond at time step {t}. AR coefficients will be NaN.")
            # At[t,:,:] will remain NaN as initialized

    print("De-snapshotting analysis complete.")
    return DeSnap_results
