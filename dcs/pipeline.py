import dataclasses
import logging
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from .causality import time_varying_causality
from .config import PipelineConfig
from .models import compute_multi_trial_BIC
from .simulation import simulate_ar_event_bootstrap
from .utils import (compute_event_statistics, desnapanalysis,
                    estimate_residuals, extract_event_snapshots, get_residuals,
                    remove_artifact_trials)
from .utils.signal import (find_best_shrinked_locs, find_peak_loc,
                           shrink_locs_resample_uniform)

logger = logging.getLogger(__name__)


def validate_inputs(original_signal: np.ndarray, detection_signal: np.ndarray, config: PipelineConfig) -> None:
    """Validate input parameters for the pipeline."""
    if not isinstance(original_signal, np.ndarray):
        logger.warning("original_signal must be a NumPy array.")
        original_signal = np.array(original_signal)
    if not isinstance(detection_signal, np.ndarray):
        logger.warning("detection_signal must be a NumPy array.")
        detection_signal = np.array(detection_signal)
    if not isinstance(config, PipelineConfig):
        raise TypeError("config must be a PipelineConfig object.")
    
    if original_signal.ndim != 2:
        raise ValueError("original_signal must be 2D (n_vars, time).")
    if detection_signal.ndim != 2 or detection_signal.shape[0] != 2:
        raise ValueError("detection_signal must be 2D with shape (2, time).")
    if original_signal.shape[1] != detection_signal.shape[1]:
        logger.warning(
            "original_signal and detection_signal must have the same time dimension length."
        )


def detect_events(detection_signal: np.ndarray, config: PipelineConfig) -> Tuple[np.ndarray, Optional[float]]:
    """Detect and align events based on the detection signal."""
    D_for_detection = detection_signal[0]
    d0_threshold = None
    
    if config.options.detection:
        logger.info("Performing event detection.")
        d0_threshold = np.nanmean(D_for_detection) + config.detection.thres_ratio * np.nanstd(D_for_detection)
        temp_locs = np.where(D_for_detection >= d0_threshold)[0]
        logger.info(f"Initial detection: {len(temp_locs)} points above threshold {d0_threshold:.2f}.")

        align_type = config.detection.align_type
        if align_type == "peak":
            locs = find_peak_loc(detection_signal[1], temp_locs, config.detection.l_extract)
            logger.info(f"Aligned to peaks, found {len(locs)} locations.")
        elif align_type == "pooled":
            if config.detection.shrink_flag:
                pool_window = int(np.ceil(config.detection.l_extract / 2))
                temp_locs_shrink = shrink_locs_resample_uniform(temp_locs, pool_window)
                locs, _ = find_best_shrinked_locs(D_for_detection, temp_locs_shrink, temp_locs)
                logger.info(f"Used pooled alignment with shrinking, found {len(locs)} locations.")
            else:
                locs = temp_locs
                logger.info(f"Used pooled alignment (no shrinking), using {len(locs)} locations.")
    else:
        logger.info("Skipping detection, using provided locations.")
        if config.detection.locs is None:
            raise ValueError("config.detection.locs cannot be None when config.options.detection is False")
        locs = np.array(config.detection.locs, dtype=int)
    
    return locs, d0_threshold


def remove_border_points(locs: np.ndarray, l_extract: int, signal_length: int) -> np.ndarray:
    """Remove event locations that are too close to signal borders."""
    original_length = len(locs)
    locs = locs[(locs >= l_extract) & (locs <= signal_length - l_extract)]
    if len(locs) < original_length:
        logger.info(f"Removed {original_length - len(locs)} locations too close to signal borders.")
    return locs


def perform_bic_selection(original_signal: np.ndarray, locs: np.ndarray, config: PipelineConfig) -> Tuple[Optional[Dict], int]:
    """Perform BIC model selection if enabled."""
    bic_outputs = None
    morder = config.bic.morder

    if config.options.bic:
        logger.info("Performing BIC model selection.")
        temp_bic_params_dict = {
            "Params": {"BIC": {"momax": config.bic.momax, "mode": config.bic.mode}},
            "EstimMode": "OLS",
        }
        
        try:
            event_snapshots_momax = extract_event_snapshots(
                original_signal,
                locs,
                config.bic.momax,
                config.bic.tau,
                config.detection.l_start,
                config.detection.l_extract,
            )
            bic_outputs = compute_multi_trial_BIC(event_snapshots_momax, temp_bic_params_dict)

            if 'mobic' in bic_outputs and bic_outputs['mobic'] is not None and len(bic_outputs['mobic']) > 1:
                selected_morder = bic_outputs['mobic'][1]
                if not np.isnan(selected_morder):
                    morder = int(selected_morder)
                    logger.info(f"BIC selected model order: {morder}")
                else:
                    logger.warning("BIC calculation resulted in NaN optimal order. Using default morder.")
            else:
                logger.error(f"Could not find 'mobic' in BIC output: {bic_outputs.keys()}")
                raise KeyError("Optimal model order key not found in BIC results.")
        except Exception as e:
            logger.error(f"BIC calculation failed: {e}. Using default morder: {morder}")
            raise RuntimeError(f"BIC calculation failed: {e}") from e

    return bic_outputs, morder


def perform_causality_analysis(event_snapshots: np.ndarray, event_stats: Dict, config: PipelineConfig) -> Optional[Dict]:
    """Perform causality analysis if enabled."""
    if not config.options.causal_analysis:
        return None

    logger.info("Performing causality analysis...")
    causal_params_dict = {
        "ref_time": config.causal.ref_time,
        "estim_mode": config.causal.estim_mode,
        "morder": event_stats.get("morder", config.bic.morder),
        "diag_flag": config.causal.diag_flag,
        "old_version": config.causal.old_version,
    }
    
    try:
        causal_output = {
            "OLS": time_varying_causality(event_snapshots, event_stats, causal_params_dict)
        }
        logger.info("Causality analysis complete.")
        return causal_output
    except Exception as e:
        logger.error(f"Causality analysis failed: {e}")
        raise


def perform_bootstrapping(event_snapshots: np.ndarray, event_stats: Dict, config: PipelineConfig) -> Optional[List[Dict]]:
    """Perform bootstrapping if enabled."""
    if not config.options.bootstrap or config.monte_carlo is None:
        return None

    logger.info(f"Starting bootstrapping ({config.monte_carlo.n_btsp} samples)...")
    bootstrap_causal_outputs_list = []
    
    simobj_dict_bootstrap = {
        "nvar": event_stats["OLS"]["At"].shape[1],
        "morder": event_stats.get("morder", config.bic.morder),
        "L": config.detection.l_extract,
        "Ntrials": event_snapshots.shape[2],
    }

    try:
        residuals_for_btsp = get_residuals(event_snapshots, event_stats)
        logger.info("Calculated residuals for bootstrapping.")
        
        for n_btsp in range(1, config.monte_carlo.n_btsp + 1):
            logger.debug(f"Calculating bootstrap trial: {n_btsp}")
            try:
                btsp_snapshots = simulate_ar_event_bootstrap(
                    simobj_dict_bootstrap, event_snapshots, event_stats, residuals_for_btsp   
                )
                btsp_stats = compute_event_statistics(btsp_snapshots, simobj_dict_bootstrap["morder"])
                btsp_causal_output = perform_causality_analysis(btsp_snapshots, btsp_stats, config)
                if btsp_causal_output:
                    bootstrap_causal_outputs_list.append(btsp_causal_output)

                if config.options.save_flag:
                    save_bootstrap_results(n_btsp, config, btsp_causal_output, btsp_snapshots, btsp_stats)
            except Exception as e:
                logger.error(f"Bootstrap trial {n_btsp} failed: {e}")
    except Exception as e:
        logger.error(f"Failed to get residuals for bootstrapping: {e}")

    return bootstrap_causal_outputs_list


def perform_desnap_analysis(original_signal: np.ndarray, detection_signal: np.ndarray, 
                          event_stats: Dict, morder: int, d0_threshold: Optional[float], 
                          config: PipelineConfig) -> Tuple[Optional[Dict], Optional[Dict]]:
    """Perform DeSnap analysis if enabled."""
    if not config.options.debiased_stats:
        return None, None

    desnap_full_output = None
    event_stats_unconditional = None
    desnap_params_instance = config.desnap

    desnap_params_instance.detection_signal = detection_signal[0]
    desnap_params_instance.original_signal = original_signal
    desnap_params_instance.Yt_stats_cond = event_stats
    desnap_params_instance.morder = morder
    desnap_params_instance.d0 = d0_threshold
    
    if desnap_params_instance.d0_max is None and desnap_params_instance.maxStdRatio is not None:
        desnap_params_instance.d0_max = np.mean(detection_signal[0]) + desnap_params_instance.maxStdRatio * np.std(detection_signal[0])
    
    try:
        desnap_full_output = desnapanalysis(desnap_params_instance)
        if 'Yt_stats_uncond' in desnap_full_output:
            event_stats_unconditional = desnap_full_output['Yt_stats_uncond']
            
            if 'OLS' not in event_stats_unconditional:
                event_stats_unconditional['OLS'] = {}
            bt_uncond, sigma_et_uncond, _ = estimate_residuals(event_stats_unconditional)
            event_stats_unconditional['OLS']['bt'] = bt_uncond
            event_stats_unconditional['OLS']['Sigma_Et'] = sigma_et_uncond
            logger.info("DeSnap analysis complete. Unconditional stats derived.")
    except Exception as e:
        logger.error(f"Desnapanalysis step failed: {e}")

    return desnap_full_output, event_stats_unconditional


def save_results(config: PipelineConfig, snap_analysis_output: Dict, event_snapshots: np.ndarray, 
                event_stats: Dict) -> None:
    """Save results to file if save_flag is enabled."""
    if not config.options.save_flag:
        return

    file_keyword = config.output.file_keyword
    outfile_final = f"{file_keyword}_model_causality.npz"
    logger.info(f"Saving final results to {outfile_final}...")
    
    event_stats_to_save = event_stats.copy()
    event_stats_to_save["mean"] = event_stats_to_save["mean"][:2, :]
    event_stats_to_save["Sigma"] = event_stats_to_save["Sigma"][:, :2, :2]
    
    try:
        config_dict_to_save = dataclasses.asdict(config)
        np.savez_compressed(
            outfile_final,
            Config=config_dict_to_save,
            SnapAnalyOutput=snap_analysis_output,
            EventSnapshots=event_snapshots,
            Yt_stats=event_stats_to_save,
        )
        logger.info("Final results saved.")
    except Exception as e:
        logger.error(f"Failed to save final results: {e}")


def save_bootstrap_results(n_btsp: int, config: PipelineConfig, btsp_causal_output: Dict, 
                         btsp_snapshots: np.ndarray, btsp_stats: Dict) -> None:
    """Save bootstrap results to file."""
    outfile_btsp = f"{config.output.file_keyword}_bootstrap_sample_{n_btsp}.npz"
    config_dict_to_save = dataclasses.asdict(config)
    np.savez_compressed(
        outfile_btsp,
        params=config_dict_to_save,
        CausalOutput_bootstrap_sample=btsp_causal_output,
        EventSnapshots=btsp_snapshots,
        Yt_stats=btsp_stats
    )
    logger.debug(f"Saved bootstrap sample {n_btsp} to {outfile_btsp}")


def remove_artifacts(event_snapshots: np.ndarray, locs: np.ndarray, config: PipelineConfig) -> Tuple[np.ndarray, np.ndarray]:
    """
    Remove artifact trials from the event snapshots.
    
    Args:
        event_snapshots (np.ndarray): Array of event snapshots
        locs (np.ndarray): Array of event locations
        config (PipelineConfig): Configuration object
        
    Returns:
        Tuple containing:
        - event_snapshots (np.ndarray): Filtered event snapshots
        - locs (np.ndarray): Updated event locations
    """
    if not config.detection.remove_artif:
        return event_snapshots, locs
        
    original_trials = event_snapshots.shape[2]
    threshold = -15000
    logger.info(f"Removing artifact trials below threshold {threshold}...")
    event_snapshots, locs_filtered = remove_artifact_trials(event_snapshots, locs, threshold)
    removed_count = original_trials - event_snapshots.shape[2]
    
    if removed_count > 0:
        logger.info(f"Removed {removed_count} artifact trials. {event_snapshots.shape[2]} trials remaining.")
        locs = locs_filtered
    else:
        logger.info("No artifact trials removed.")
    
    return event_snapshots, locs


def extract_event_snapshots_with_config(original_signal: np.ndarray, locs: np.ndarray, 
                                      morder: int, config: PipelineConfig) -> np.ndarray:
    """
    Extract event snapshots using configuration parameters.
    
    Args:
        original_signal (np.ndarray): Original time series data
        locs (np.ndarray): Array of event locations
        morder (int): Model order to use
        config (PipelineConfig): Configuration object
        
    Returns:
        np.ndarray: Extracted event snapshots
    """
    final_tau = config.bic.tau if config.options.bic else 1
    logger.info(f"Extracting final event snapshots (morder={morder}, tau={final_tau})...")
    
    event_snapshots = extract_event_snapshots(
        original_signal, 
        locs, 
        morder, 
        final_tau, 
        config.detection.l_start, 
        config.detection.l_extract
    )
    
    return event_snapshots


def compute_event_statistics_with_error_handling(event_snapshots: np.ndarray, morder: int) -> Dict:
    """
    Compute event statistics with proper error handling.
    
    Args:
        event_snapshots (np.ndarray): Array of event snapshots
        morder (int): Model order to use
        
    Returns:
        Dict: Computed event statistics
        
    Raises:
        Exception: If statistics computation fails
    """
    logger.info("Computing conditional event statistics...")
    try:
        event_stats = compute_event_statistics(event_snapshots, morder)
        return event_stats
    except Exception as e:
        logger.error(f"Failed to compute event statistics: {e}")
        raise


def prepare_final_output(d0_threshold: Optional[float], locs: np.ndarray, morder: int,
                        event_stats: Dict, causal_output: Optional[Dict],
                        bic_outputs: Optional[Dict], bootstrap_causal_outputs_list: Optional[List[Dict]],
                        desnap_full_output: Optional[Dict], event_stats_unconditional: Optional[Dict]) -> Dict:
    """
    Prepare the final output dictionary with all analysis results.
    
    Args:
        d0_threshold (Optional[float]): Detection threshold used
        locs (np.ndarray): Detected event locations
        morder (int): Model order used
        event_stats (Dict): Event statistics
        causal_output (Optional[Dict]): Causality analysis results
        bic_outputs (Optional[Dict]): BIC selection results
        bootstrap_causal_outputs_list (Optional[List[Dict]]): Bootstrap results
        desnap_full_output (Optional[Dict]): DeSnap analysis results
        event_stats_unconditional (Optional[Dict]): Unconditional statistics
        
    Returns:
        Dict: Complete analysis output
    """
    return {
        "d0": d0_threshold,
        "locs": locs,
        "morder": morder,
        "Yt_stats": event_stats,
        "CausalOutput": causal_output,
        "BICoutputs": bic_outputs,
        "BootstrapCausalOutputs": bootstrap_causal_outputs_list,
        "DeSnap_output": desnap_full_output,
        "Yt_stats_unconditional": event_stats_unconditional
    }


def snapshot_detect_analysis_pipeline(
    original_signal: np.ndarray,
    detection_signal: np.ndarray,
    config: PipelineConfig,
) -> Tuple[Dict, PipelineConfig, np.ndarray]:
    """
    Perform a pipeline for detecting and analyzing events in time series data.

    This function orchestrates event detection, optional BIC model selection,
    event snapshot extraction, artifact removal, statistics computation,
    causality analysis, and optional bootstrapping based on the provided
    configuration object.

    Args:
        original_signal (np.ndarray): Original time series data, expected shape
            (n_vars, n_time_points). Often trial-averaged data.
        detection_signal (np.ndarray): Signal used for event detection and
            alignment, expected shape (2, n_time_points). Channel 0 is often
            used for thresholding, Channel 1 for alignment (e.g., peak finding).
        config (PipelineConfig): Configuration object containing all parameters.

    Returns:
        Tuple containing:
        - SnapAnalyOutput (dict): Analysis results
        - config (PipelineConfig): Final configuration
        - Yt_events (np.ndarray): Extracted snapshot array
    """
    # Validate inputs
    validate_inputs(original_signal, detection_signal, config)
    
    snap_analysis_output: Dict[str, Any] = {}
    logger.info("Starting snapshot detection and analysis pipeline.")

    # Detect events
    locs, d0_threshold = detect_events(detection_signal, config)
    
    # Remove border points
    locs = remove_border_points(locs, config.detection.l_extract, original_signal.shape[1])
    if len(locs) == 0:
        logger.warning("No valid event locations remaining after border removal.")
        snap_analysis_output.update({"locs": locs, "morder": config.bic.morder})
        return snap_analysis_output, config, np.array([])

    # Perform BIC selection
    bic_outputs, morder = perform_bic_selection(original_signal, locs, config)

    # Extract event snapshots
    event_snapshots = extract_event_snapshots_with_config(original_signal, locs, morder, config)
    if event_snapshots.shape[2] == 0:
        logger.warning("No trials available for final analysis after snapshot extraction.")
        snap_analysis_output.update({"locs": locs, "morder": morder, "d0": d0_threshold})
        return snap_analysis_output, config, event_snapshots

    # Remove artifacts if enabled
    event_snapshots, locs = remove_artifacts(event_snapshots, locs, config)
    if event_snapshots.shape[2] == 0:
        logger.warning("No trials remaining after artifact removal.")
        snap_analysis_output.update({"locs": locs, "morder": morder, "d0": d0_threshold})
        return snap_analysis_output, config, event_snapshots

    # Compute statistics
    event_stats = compute_event_statistics_with_error_handling(event_snapshots, morder)

    # Perform causality analysis
    causal_output = perform_causality_analysis(event_snapshots, event_stats, config)

    # Perform bootstrapping
    bootstrap_causal_outputs_list = perform_bootstrapping(event_snapshots, event_stats, config)

    # Perform DeSnap analysis
    desnap_full_output, event_stats_unconditional = perform_desnap_analysis(
        original_signal, detection_signal, event_stats, morder, d0_threshold, config
    )

    # Prepare final output
    snap_analysis_output = prepare_final_output(
        d0_threshold, locs, morder, event_stats, causal_output,
        bic_outputs, bootstrap_causal_outputs_list,
        desnap_full_output, event_stats_unconditional
    )

    # Save results if enabled
    save_results(config, snap_analysis_output, event_snapshots, event_stats)

    logger.info("Snapshot detection and analysis pipeline finished.")
    return snap_analysis_output, config, event_snapshots
