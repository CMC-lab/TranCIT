import logging
from typing import Dict, Tuple

import numpy as np
from utils.residuals import get_residuals
from utils.timefreq import compute_simulated_timefreq

from .causality import time_varying_causality
from .models import compute_multi_trial_BIC
from .simulation import simulate_ar_event_bootstrap
from .utils.core import compute_event_statistics, extract_event_snapshots
from .utils.preprocess import remove_artifact_trials
from .utils.signal import (
    find_best_shrinked_locs,
    find_peak_loc,
    shrink_locs_resample_uniform,
)

logger = logging.getLogger(__name__)


def snapshot_detect_analysis_pipeline(
    original_signal: np.ndarray, detection_signal: np.ndarray, params: Dict
) -> Tuple[Dict, Dict, np.ndarray]:
    """
    Perform a pipeline for detecting and analyzing events in time series data.

    This function detects reference points, extracts event snapshots, and optionally performs
    BIC model selection, causality analysis, and bootstrapping based on
    the provided parameters.

    Args:
        original_signal (np.ndarray): Original time series, shape (n_vars, n_time_points).
        detection_signal (np.ndarray): Signal for event detection, shape (2, n_time_points).
        params (Dict): Configuration dictionary with nested keys specifying pipeline behavior:
            - 'Options' (Dict): Flags for enabling steps:
                - 'Detection' (int): 1 to perform detection, 0 to use provided locs.
                - 'BIC' (bool): True to perform BIC model selection.
                - 'CausalAnalysis' (bool): True to compute causality measures.
                - 'Bootstrap' (bool): True to perform bootstrapping for causality.
                - 'save_flag' (bool): True to save outputs to files.
            - 'Detection' (Dict): Parameters for event detection:
                - 'ThresRatio' (float): Threshold ratio for peak detection relative to std dev.
                - 'AlignType' (str): 'peak' or 'pooled' for aligning events.
                - 'L_extract' (int): Length of the window to extract around each event.
                - 'L_start' (int): Offset from the detected location to start extraction.
                - 'ShrinkFlag' (bool): Whether to use shrinking for 'pooled' alignment.
                - 'locs' (np.ndarray, optional): Predefined locations if Options['Detection']==0.
                - 'remove_artif' (bool, optional): True to remove artifact trials based on threshold.
            - 'BIC' (Dict): Parameters for Bayesian Information Criterion:
                - 'momax' (int): Maximum model order to test.
                - 'tau' (int): Lag step used for constructing event snapshots for BIC.
                - 'morder' (int): Default model order if BIC is not run.
                - 'mode' (str): BIC calculation mode (e.g., 'biased').
            - 'CausalParams' (Dict): Parameters for causality analysis (passed to time_varying_causality).
                - Includes 'ref_time', 'estim_mode', 'diag_flag', 'old_version'.
            - 'MonteC_Params' (Dict): Parameters for bootstrapping.
                - Includes 'Nbtsp' (number of bootstrap samples).
            - 'Output' (Dict): Settings for saving outputs.
                - 'FileKeyword' (str): Base keyword for output filenames.

    Returns:
        Tuple[Dict, Dict, np.ndarray]:
            - SnapAnalyOutput: Dictionary containing analysis results like 'locs', 'morder', 'Yt_stats', 'CausalOutput', 'BICoutputs'.
            - params: Updated parameters dictionary (may include added info like 'DeSnap_inputs').
            - Yt_events: Extracted event snapshots, shape (n_vars * (model_order + 1), L_extract, n_trials).

    Raises:
        KeyError: If required keys are missing in params.
        ValueError: If parameters like 'AlignType' or 'BIC['mode']' are invalid.
    """
    # --- Start: Added Parameter Checks ---
    # Validate top-level keys
    required_keys = [
        "Options",
        "Detection",
        "BIC",
        "CausalParams",
        "MonteC_Params",
        "PSD",
        "Output",
    ]
    for key in required_keys:
        if key not in params:
            raise KeyError(
                f"Missing required top-level key '{key}' in params dictionary."
            )

    # Validate necessary sub-keys based on options
    options = params["Options"]  # Safe to access now
    if not isinstance(options, dict):
        raise TypeError("params['Options'] must be a dictionary.")

    detection_params = params["Detection"]  # Safe to access now
    if not isinstance(detection_params, dict):
        raise TypeError("params['Detection'] must be a dictionary.")
    if options.get("Detection", 0) == 1:  # Check sub-keys only if detection is active
        req_detect_keys = [
            "ThresRatio",
            "AlignType",
            "L_extract",
            "L_start",
            "ShrinkFlag",
        ]
        for key in req_detect_keys:
            if key not in detection_params:
                raise KeyError(
                    f"Missing required detection parameter: params['Detection']['{key}']"
                )
        align_type = detection_params["AlignType"]  # Safe to access
        if align_type not in ["peak", "pooled"]:
            raise ValueError(f"Invalid AlignType '{align_type}' in params['Detection']")
    else:  # If detection is off, locs must be provided
        if "locs" not in detection_params:
            raise KeyError(
                "Missing required parameter: params['Detection']['locs'] when Options['Detection'] is 0."
            )

    bic_params = params["BIC"]  # Safe to access
    if not isinstance(bic_params, dict):
        raise TypeError("params['BIC'] must be a dictionary.")
    if options.get("BIC", False):  # Check sub-keys if BIC is active
        req_bic_keys = ["momax", "tau", "mode"]
        for key in req_bic_keys:
            if key not in bic_params:
                raise KeyError(
                    f"Missing required BIC parameter: params['BIC']['{key}']"
                )
    if "morder" not in bic_params:  # morder is needed even if BIC is off
        raise KeyError("Missing required BIC parameter: params['BIC']['morder']")

    if "FileKeyword" not in params["Output"]:
        raise KeyError(
            "Missing required Output parameter: params['Output']['FileKeyword']"
        )
    # --- End: Added Parameter Checks ---

    snap_analysis_output = {}

    # Step 1: Detect reference points
    if params["Options"]["Detection"] == 1:
        D = detection_signal[0]
        d0 = params["Detection"].get(
            "d0", np.nanmean(D) + params["Detection"]["ThresRatio"] * np.nanstd(D)
        )
        temp_locs = np.where(D >= d0)[0]

        align_type = params["Detection"]["AlignType"]
        if align_type == "peak":
            locs = find_peak_loc(
                detection_signal[1], temp_locs, params["Detection"]["L_extract"]
            )
        elif align_type == "pooled":
            if params["Detection"]["ShrinkFlag"]:
                locs = shrink_locs_resample_uniform(
                    temp_locs, int(np.ceil(params["Detection"]["L_extract"] / 2))
                )
                locs, _ = find_best_shrinked_locs(D, locs, temp_locs)
            else:
                locs = temp_locs
        else:
            raise ValueError(f"Invalid AlignType: {align_type}")
    else:
        locs = params["Detection"]["locs"]

    # Step 2: Remove border points
    L_extract = params["Detection"]["L_extract"]
    locs = locs[(locs >= L_extract) & (locs <= original_signal.shape[1] - L_extract)]

    # Step 3: BIC model selection (optional)
    if params["Options"]["BIC"]:
        logger.info("Performing BIC model selection")
        bic_parser = {
            "OriSignal": original_signal,
            "DetSignal": detection_signal,
            "Params": params,
            "EstimMode": "OLS",
        }
        if params["BIC"]["mode"] == "biased":
            event_snapshots_momax = extract_event_snapshots(
                original_signal,
                locs,
                params["BIC"]["momax"],
                params["BIC"]["tau"],
                params["Detection"]["L_start"],
                L_extract,
            )
            bic_outputs = compute_multi_trial_BIC(event_snapshots_momax, bic_parser)
            morder = bic_outputs["mobic"][1]
        else:
            raise ValueError(f"Unsupported BIC mode: {params['BIC']['mode']}")
        np.savez(
            f"{params['Output']['FileKeyword']}_BIC.npz",
            Params=params,
            BICoutputs=bic_outputs,
        )
    else:
        morder = params["BIC"]["morder"]
        bic_outputs = None

    # Step 4: Extract event snapshots
    event_snapshots = extract_event_snapshots(
        original_signal,
        locs,
        morder,
        params["BIC"]["tau"],
        params["Detection"]["L_start"],
        L_extract,
    )

    # Step 5: Remove artifacts
    if params["Detection"].get("remove_artif", False):
        event_snapshots, locs = remove_artifact_trials(event_snapshots, locs, -15000)

    # Step 6: Compute statistics
    event_stats = compute_event_statistics(event_snapshots, morder)

    # Step 7: Causality analysis
    causal_output = None
    if params["Options"].get("CausalAnalysis", False):
        causal_params = params["CausalParams"].copy()
        causal_params["morder"] = morder
        causal_output = {
            "OLS": time_varying_causality(event_snapshots, event_stats, causal_params)
        }

    # Step 8: Bootstrapping
    if params["Options"].get("Bootstrap", False):
        logger.info("Starting bootstrapping")
        monte_c_params = params["MonteC_Params"].copy()
        monte_c_params["morder"] = morder
        residuals = get_residuals(event_snapshots, event_stats)
        for n_btsp in range(1, monte_c_params["Nbtsp"] + 1):
            logger.info(f"Calculating bootstrap trial: {n_btsp}")
            btsp_snapshots = simulate_ar_event_bootstrap(
                monte_c_params, event_snapshots, event_stats, residuals
            )
            btsp_stats = compute_event_statistics(btsp_snapshots, morder)
            btsp_causal_output = {
                "OLS": time_varying_causality(btsp_snapshots, btsp_stats, causal_params)
            }
            np.savez_compressed(
                f"{params['Output']['FileKeyword']}_btsp_{n_btsp}_model_causality.npz",
                Params=params,
                CausalOutput_btsp=btsp_causal_output,
                Yt_stats_btsp=btsp_stats,
            )

    # Step 10: Prepare and save results
    snap_analysis_output.update(
        {
            "d0": d0 if params["Options"]["Detection"] == 1 else None,
            "locs": locs,
            "morder": morder,
            "Yt_stats": event_stats,
            "CausalOutput": causal_output,
            "BICoutputs": bic_outputs,
        }
    )

    if params["Options"].get("save_flag", False):
        event_stats["mean"] = event_stats["mean"][:2, :]
        event_stats["Sigma"] = event_stats["Sigma"][:, :2, :2]
        params["DeSnap_inputs"] = {
            "x": [],
            "y": original_signal,
            "yf": detection_signal,
            "D": snap_analysis_output["d0"],
            "Yt_stats_cond": event_stats,
        }
        file_keyword = params["Output"]["FileKeyword"]
        if params["Options"].get("CausalAnalysis", False):
            np.savez_compressed(
                f"{file_keyword}_model_causality.npz",
                Params=params,
                Yt_stats=event_stats,
                CausalOutput=causal_output,
                SnapAnalyOutput=snap_analysis_output,
            )
        else:
            np.savez_compressed(
                f"{file_keyword}_model.npz", Params=params, Yt_stats=event_stats
            )

    return snap_analysis_output, params, event_snapshots
