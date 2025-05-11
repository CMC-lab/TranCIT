import dataclasses
import logging
from typing import Dict, Tuple, Any, Optional

import numpy as np

from .causality import time_varying_causality
from .config import DeSnapParams, PipelineConfig
from .models import compute_multi_trial_BIC
from .simulation import simulate_ar_event_bootstrap
from .utils import (compute_event_statistics, desnapanalysis,
                    estimate_residuals, extract_event_snapshots, get_residuals,
                    remove_artifact_trials)
from .utils.signal import (find_best_shrinked_locs, find_peak_loc,
                           shrink_locs_resample_uniform)

logger = logging.getLogger(__name__)


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
        config (PipelineConfig): A dataclass object containing all configuration
            parameters for the pipeline. Includes nested dataclasses for options
            (config.options - including 'debiased_stats'), detection (config.detection),
            BIC (config.bic), causality (config.causal), bootstrapping (config.monte_carlo),
            DeSnap settings (config.desnap_settings), and output (config.output).
            See PipelineConfig and related dataclass definitions for details.

    Returns:
        A tuple containing:

        - **SnapAnalyOutput (dict):** Contains keys:
            * `'d0'` (float or None): Detection threshold used, if calculated.
            * `'locs'` (np.ndarray): Detected and filtered event locations.
            * `'morder'` (int): The VAR model order used.
            * `'Yt_stats'` (dict): Snapshot statistics (mean, cov, coeffs).
            * `'CausalOutput'` (dict or None): DCS/TE/rDCS results if available.
            * `'BICoutputs'` (dict or None): BIC selection outputs, if enabled.
            * `DeSnap_output` (Dict | None): Full output from the desnapanalysis function, if run.
            * `event_stats_debiased` (Dict | None): Unconditional statistics derived from DeSnap analysis, if run.

        - **config (PipelineConfig):** Final configuration object.

        - **Yt_events (np.ndarray):** Extracted snapshot array of shape
        *(n_vars × (model_order + 1), L_extract, n_trials_filtered)*.

    :rtype: Tuple[Dict[str, Any], PipelineConfig, np.ndarray]

    Raises:
        ValueError: If configuration parameters are inconsistent (e.g., detection
            is off but no locs provided, align_type invalid). Checked in PipelineConfig.__post_init__.
        TypeError: If input signals are not NumPy arrays or config is not PipelineConfig.
        KeyError: If essential results are unexpectedly missing during internal steps
             (should be less likely with structured config).
        Exception: Can propagate exceptions from underlying functions (e.g., linalg errors).
    """
    # --- Input Validation ---
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

    snap_analysis_output: Dict[str, Any] = {}
    logger.info("Starting snapshot detection and analysis pipeline.")

    d0_threshold: Optional[float] = None
    l_extract = config.detection.l_extract
    
    # --- Step 1: Detect reference points ---
    D_for_detection = detection_signal[0]
    if config.options.detection:
        logger.info("Performing event detection.")
        d0_threshold = np.nanmean(D_for_detection) + config.detection.thres_ratio * np.nanstd(D_for_detection)
        temp_locs = np.where(D_for_detection >= d0_threshold)[0]
        logger.info(
            f"Initial detection: {len(temp_locs)} points above threshold {d0_threshold:.2f}."
        )

        align_type = config.detection.align_type
        if align_type == "peak":
            locs = find_peak_loc(detection_signal[1], temp_locs, l_extract)
            logger.info(f"Aligned to peaks, found {len(locs)} locations.")
        elif align_type == "pooled":
            if config.detection.shrink_flag:
                pool_window = int(np.ceil(l_extract / 2))
                temp_locs_shrink = shrink_locs_resample_uniform(temp_locs, pool_window)
                locs, _ = find_best_shrinked_locs(D_for_detection, temp_locs_shrink, temp_locs)
                logger.info(
                    f"Used pooled alignment with shrinking, found {len(locs)} locations."
                )
            else:
                locs = temp_locs
                logger.info(
                    f"Used pooled alignment (no shrinking), using {len(locs)} locations."
                )
    else:
        logger.info("Skipping detection, using provided locations.")
        if config.detection.locs is None:
            raise ValueError(
                "config.detection.locs cannot be None when config.options.detection is False"
            )
        locs = np.array(config.detection.locs, dtype=int)

    # --- Step 2: Remove border points ---
    original_length = len(locs)
    locs = locs[(locs >= l_extract) & (locs <= original_signal.shape[1] - l_extract)]
    if len(locs) < original_length:
        logger.info(
            f"Removed {original_length - len(locs)} locations too close to signal borders."
        )
    if len(locs) == 0:
        logger.warning("No valid event locations remaining after border removal.")
        snap_analysis_output.update({"locs": locs, "morder": config.bic.morder})
        return snap_analysis_output, config, np.array([])

    # --- Step 3: BIC model selection (optional) ---
    bic_outputs = None
    morder = config.bic.morder  # Get base model order

    if config.options.bic:
        logger.info("Performing BIC model selection.")
        temp_bic_params_dict = {
            "Params": {"BIC": {"momax": config.bic.momax, "mode": config.bic.mode}},
            "EstimMode": "OLS",
        }
        logger.info(
            f"Extracting snapshots for BIC (momax={config.bic.momax}, tau={config.bic.tau})..."
        )
        try:
            event_snapshots_momax = extract_event_snapshots(
                original_signal,
                locs,
                config.bic.momax,
                config.bic.tau,
                config.detection.l_start,
                l_extract,
            )
            logger.info("Running compute_multi_trial_BIC...")
            bic_outputs = compute_multi_trial_BIC(
                event_snapshots_momax, temp_bic_params_dict
            )

            selected_morder = bic_outputs["mobic"][1]
            if 'mobic' in bic_outputs and bic_outputs['mobic'] is not None and len(bic_outputs['mobic']) > 1:
                selected_morder = bic_outputs['mobic'][1]
            else:
                logger.error(f"Could not find 'mobic' in BIC output: {bic_outputs.keys()}")
                raise KeyError("Optimal model order key not found in BIC results.")

            if np.isnan(selected_morder):
                logger.warning(
                    "BIC calculation resulted in NaN optimal order. Using default morder."
                )
            else:
                morder = int(selected_morder)
                logger.info(f"BIC selected model order: {morder}")
        except Exception as e:
            logger.error(f"BIC calculation failed: {e}. Using default morder: {morder}")
            raise RuntimeError(f"BIC calculation failed: {e}") from e

    # --- Step 4: Extract event snapshots with final morder ---
    final_tau = config.bic.tau if config.options.bic else 1
    logger.info(
        f"Extracting final event snapshots (morder={morder}, tau={final_tau})..."
    )
    event_snapshots = extract_event_snapshots(
        original_signal, locs, morder, final_tau, config.detection.l_start, l_extract
    )
    if event_snapshots.shape[2] == 0:
        logger.warning("No trials available for final analysis after snapshot extraction.")
        snap_analysis_output.update({"locs": locs, "morder": morder, "d0": d0_threshold})
        return snap_analysis_output, config, event_snapshots
    
    logger.info(f"Extracted snapshots shape: {event_snapshots.shape}")

    # --- Step 5: Remove artifacts ---
    if config.detection.remove_artif:
        original_trials = event_snapshots.shape[2]
        threshold = -15000
        logger.info(f"Removing artifact trials below threshold {threshold}...")
        event_snapshots, locs_filtered = remove_artifact_trials(
            event_snapshots, locs, threshold
        )
        removed_count = original_trials - event_snapshots.shape[2]
        if removed_count > 0:
            logger.info(
                f"Removed {removed_count} artifact trials. {event_snapshots.shape[2]} trials remaining."
            )
            locs = locs_filtered
        else:
            logger.info("No artifact trials removed.")
        if event_snapshots.shape[2] == 0:
            logger.warning("No trials remaining after artifact removal.")
            snap_analysis_output.update({"locs": locs, "morder": morder, "d0": d0_threshold})
            return (
                snap_analysis_output,
                config,
                event_snapshots,
            )  # Return empty snapshots

    # --- Step 6: Compute statistics ---
    logger.info("Computing conditional event statistics...")
    try:
        event_stats = compute_event_statistics(event_snapshots, morder)
    except Exception as e:
        logger.error(f"Failed to compute event statistics: {e}")
        raise

    # --- Step 7: Causality analysis ---
    causal_output = None
    if config.options.causal_analysis:
        logger.info("Performing causality analysis...")
        causal_params_dict = {
            "ref_time": config.causal.ref_time,
            "estim_mode": config.causal.estim_mode,
            "morder": morder,
            "diag_flag": config.causal.diag_flag,
            "old_version": config.causal.old_version,
        }
        try:
            causal_output = {
                "OLS": time_varying_causality(
                    event_snapshots, event_stats, causal_params_dict
                )
            }
            logger.info("Causality analysis complete.")
        except Exception as e:
            logger.error(f"Causality analysis failed: {e}")
            raise

    # --- Step 8: Bootstrapping ---
    bootstrap_causal_outputs_list = None
    if config.options.bootstrap:
        if config.monte_carlo is None:
            logger.warning("Bootstrap requested but no Monte Carlo parameters provided. Skipping bootstrap.")
        else:
            logger.info(f"Starting bootstrapping ({config.monte_carlo.n_btsp} samples)...")
            simobj_dict_bootstrap = {
                "nvar": event_stats["OLS"]["At"].shape[1],
                "morder": morder,
                "L": l_extract,
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
                        btsp_stats = compute_event_statistics(btsp_snapshots, morder)
                        btsp_causal_output = {
                            "OLS": time_varying_causality(
                                btsp_snapshots, btsp_stats, causal_params_dict
                            )
                        }
                        bootstrap_causal_outputs_list.append(btsp_causal_output)

                        if config.options.save_flag:
                            outfile_btsp = f"{config.output.file_keyword}_bootstrap_sample_{n_btsp}.npz"
                            config_dict_to_save = dataclasses.asdict(config)
                            np.savez_compressed(
                                outfile_btsp,
                                params=config_dict_to_save,
                                CausalOutput_bootstrap_sample=bootstrap_causal_outputs_list,
                                EventSnapshots=btsp_snapshots,
                                Yt_stats=btsp_stats
                            )
                            logger.debug(f"Saved bootstrap sample {n_btsp} to {outfile_btsp}")
                    except Exception as e:
                        logger.error(f"Bootstrap trial {n_btsp} failed: {e}")
            except Exception as e:
                logger.error(f"Failed to get residuals for bootstrapping: {e}")

    # --- Step 10: Perform DeSnap (Derive Unconditional Statistics) ---
    desnap_full_output = None
    event_stats_unconditional = None
    
    if config.options.debiased_stats:
        if config.desnap_settings is None:
            logger.warning("DeSnap (debiased_stats) analysis requested but no desnapping settings (config.desnap_settings) provided. Skipping DeSnap.")
        elif d0_threshold is None and config.desnap_settings.maxStdRatio is None and config.desnap_settings.d0_max_val is None :
             logger.warning("DeSnap analysis requires either a d0 from detection or d0_max/maxStdRatio in desnapping_settings. Skipping DeSnap.")
        elif event_stats is None:
            logger.warning("Event statistics (event_stats) not available. Skipping DeSnap.")
        else:
            logger.info("Performing DeSnap analysis to derive unconditional statistics...")
            
        logger.info("Performing DeSnap analysis...")
        desnap_params_instance = DeSnapParams(
            detection_signal=D_for_detection,
            original_signal=original_signal,
            Yt_stats_cond=event_stats,
            morder=morder,
            tau=config.get('DeSnap_inputs', {}).get('tau', None),
            l_start=config.get('DeSnap_inputs', {}).get('l_start', None),
            l_extract=config.get('DeSnap_inputs', {}).get('l_extract', None),
            d0=d0_threshold,
            N_d=config.get('DeSnap_inputs', {}).get('N_d', 10),
            d0_max=config.get('DeSnap_inputs', {}).get('d0_max', None),
            maxStdRatio=config.get('DeSnap_inputs', {}).get('maxStdRatio', None),
            diff_flag=config.get('DeSnap_inputs', {}).get('diff_flag', False)
        )
        if desnap_params_instance.d0_max is None and desnap_params_instance.maxStdRatio is not None:
            desnap_params_instance.d0_max = np.mean(D_for_detection) + desnap_params_instance.maxStdRatio * np.std(D_for_detection)
        
        try:
            desnap_full_output = desnapanalysis(desnap_params_instance)
            if 'Yt_stats_uncond' in desnap_full_output:
                event_stats_unconditional = desnap_full_output['Yt_stats_uncond']
                
                if 'OLS' not in event_stats_unconditional: event_stats_unconditional['OLS'] = {}
                bt_uncond, sigma_et_uncond = estimate_residuals(event_stats_unconditional)
                event_stats_unconditional['OLS']['bt'] = bt_uncond
                event_stats_unconditional['OLS']['Sigma_Et'] = sigma_et_uncond
                logger.info("DeSnap analysis complete. Unconditional stats derived.")
            else:
                logger.warning("DeSnap analysis ran but 'Yt_stats_uncond' not found in output.")
        except Exception as e:
            logger.error(f"Desnapanalysis step failed: {e}")
    
    # --- Step 11: Prepare final output and save ---
    snap_analysis_output.update(
        {
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
    )

    if config.options.save_flag:
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

    logger.info("Snapshot detection and analysis pipeline finished.")
    return snap_analysis_output, config, event_snapshots
