import logging
from typing import Dict, Tuple

import numpy as np
from utils.residuals import get_residuals
from utils.timefreq import compute_simulated_timefreq

from .causality import time_varying_causality
from .models import compute_multi_trial_BIC
from .simulation import simulate_ar_event, simulate_ar_event_bootstrap
from .utils.core import compute_event_statistics, extract_event_snapshots
from .utils.preprocess import remove_artifact_trials
from .utils.signal import (find_best_shrinked_locs, find_peak_loc,
                           shrink_locs_resample_uniform)

logger = logging.getLogger(__name__)

def snapshot_detect_analysis_pipeline(
    original_signal: np.ndarray,
    detection_signal: np.ndarray,
    params: Dict
) -> Tuple[Dict, Dict, np.ndarray]:
    """
    Perform a pipeline for detecting and analyzing events in time series data.

    This function detects reference points, extracts event snapshots, and optionally performs
    BIC model selection, causality analysis, bootstrapping, and PSD computation.

    Args:
        original_signal (np.ndarray): Original time series, shape (n_vars, n_time_points).
        detection_signal (np.ndarray): Signal for event detection, shape (2, n_time_points).
        params (Dict): Configuration dictionary with nested keys:
            - 'Options': Flags for 'Detection', 'BIC', 'CausalAnalysis', 'Bootstrap', 'PSD', 'save_flag'.
            - 'Detection': Parameters like 'ThresRatio', 'AlignType', 'L_extract', 'L_start', 'ShrinkFlag'.
            - 'BIC': Parameters like 'momax', 'tau', 'morder', 'mode'.
            - 'CausalParams': Causality analysis settings.
            - 'MonteC_Params': Bootstrapping settings.
            - 'PSD': PSD computation settings.
            - 'Output': File saving settings.

    Returns:
        Tuple[Dict, Dict, np.ndarray]:
            - SnapAnalyOutput: Results dictionary.
            - params: Updated parameters.
            - Yt_events: Event snapshots.

    Raises:
        KeyError: If required keys are missing in params.
        ValueError: If parameters are invalid.
    """
    # Validate inputs
    required_keys = ['Options', 'Detection', 'BIC', 'Output']
    for key in required_keys:
        if key not in params:
            raise KeyError(f"Missing required key '{key}' in params")

    snap_analysis_output = {}

    # Step 1: Detect reference points
    if params['Options']['Detection'] == 1:
        D = detection_signal[0]
        d0 = params['Detection'].get('d0', np.nanmean(D) + params['Detection']['ThresRatio'] * np.nanstd(D))
        temp_locs = np.where(D >= d0)[0]

        align_type = params['Detection']['AlignType']
        if align_type == "peak":
            locs = find_peak_loc(detection_signal[1], temp_locs, params['Detection']['L_extract'])
        elif align_type == "pooled":
            if params['Detection']['ShrinkFlag']:
                locs = shrink_locs_resample_uniform(temp_locs, int(np.ceil(params['Detection']['L_extract'] / 2)))
                locs, _ = find_best_shrinked_locs(D, locs, temp_locs)
            else:
                locs = temp_locs
        else:
            raise ValueError(f"Invalid AlignType: {align_type}")
    else:
        locs = params['Detection']['locs']

    # Step 2: Remove border points
    L_extract = params['Detection']['L_extract']
    locs = locs[(locs >= L_extract) & (locs <= original_signal.shape[1] - L_extract)]

    # Step 3: BIC model selection (optional)
    if params['Options']['BIC']:
        logger.info("Performing BIC model selection")
        bic_parser = {
            'OriSignal': original_signal,
            'DetSignal': detection_signal,
            'Params': params,
            'EstimMode': 'OLS'
        }
        if params['BIC']['mode'] == 'biased':
            event_snapshots_momax = extract_event_snapshots(
                original_signal, locs, params['BIC']['momax'], params['BIC']['tau'],
                params['Detection']['L_start'], L_extract
            )
            bic_outputs = compute_multi_trial_BIC(event_snapshots_momax, bic_parser)
            morder = bic_outputs['mobic'][1]
        else:
            raise ValueError(f"Unsupported BIC mode: {params['BIC']['mode']}")
        np.savez(f"{params['Output']['FileKeyword']}_BIC.npz", Params=params, BICoutputs=bic_outputs)
    else:
        morder = params['BIC']['morder']
        bic_outputs = None

    # Step 4: Extract event snapshots
    event_snapshots = extract_event_snapshots(
        original_signal, locs, morder, params['BIC']['tau'], params['Detection']['L_start'], L_extract
    )

    # Step 5: Remove artifacts
    if params['Detection'].get('remove_artif', False):
        event_snapshots, locs = remove_artifact_trials(event_snapshots, locs, -15000)

    # Step 6: Compute statistics
    event_stats = compute_event_statistics(event_snapshots, morder)

    # Step 7: Causality analysis
    causal_output = None
    if params["Options"].get("CausalAnalysis", False):
        causal_params = params['CausalParams'].copy()
        causal_params['morder'] = morder
        causal_output = {'OLS': time_varying_causality(event_snapshots, event_stats, causal_params)}

    # Step 8: Bootstrapping
    if params['Options'].get('Bootstrap', False):
        logger.info("Starting bootstrapping")
        monte_c_params = params['MonteC_Params'].copy()
        monte_c_params['morder'] = morder
        residuals = get_residuals(event_snapshots, event_stats)
        for n_btsp in range(1, monte_c_params['Nbtsp'] + 1):
            logger.info(f"Calculating bootstrap trial: {n_btsp}")
            btsp_snapshots = simulate_ar_event_bootstrap(monte_c_params, event_snapshots, event_stats, residuals)
            btsp_stats = compute_event_statistics(btsp_snapshots, morder)
            btsp_causal_output = {'OLS': time_varying_causality(btsp_snapshots, btsp_stats, causal_params)}
            np.savez_compressed(
                f"{params['Output']['FileKeyword']}_btsp_{n_btsp}_model_causality.npz",
                Params=params, CausalOutput_btsp=btsp_causal_output, Yt_stats_btsp=btsp_stats
            )

    # Step 9: PSD calculation
    if params['Options'].get('PSD', False):
        if not params['PSD'].get('MonteC_flag', False):
            event_stats = compute_simulated_timefreq(event_snapshots, event_stats, params['PSD'])
        else:
            psd_params = params['PSD'].copy()
            psd_params['simobj']['morder'] = morder
            mc_snapshots = simulate_ar_event(psd_params['simobj'], event_stats)
            event_stats = compute_simulated_timefreq(mc_snapshots, event_stats, psd_params)

    # Step 10: Prepare and save results
    snap_analysis_output.update({
        "d0": d0 if params["Options"]["Detection"] == 1 else None,
        "locs": locs,
        "morder": morder,
        "Yt_stats": event_stats,
        "CausalOutput": causal_output,
        "BICoutputs": bic_outputs
    })

    if params['Options'].get('save_flag', False):
        event_stats['mean'] = event_stats['mean'][:2, :]
        event_stats['Sigma'] = event_stats['Sigma'][:, :2, :2]
        params['DeSnap_inputs'] = {
            'x': [], 'y': original_signal, 'yf': detection_signal,
            'D': snap_analysis_output["d0"], 'Yt_stats_cond': event_stats
        }
        file_keyword = params['Output']['FileKeyword']
        if params['Options'].get('PSD', False):
            np.savez_compressed(f"{file_keyword}_psd.npz", Params=params, PSD=event_stats['spectr'])
        elif params["Options"].get("CausalAnalysis", False):
            np.savez_compressed(f"{file_keyword}_model_causality.npz", Params=params, Yt_stats=event_stats,
                                CausalOutput=causal_output, SnapAnalyOutput=snap_analysis_output)
        else:
            np.savez_compressed(f"{file_keyword}_model.npz", Params=params, Yt_stats=event_stats)

    return snap_analysis_output, params, event_snapshots

