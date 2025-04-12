import logging
from typing import Dict, Tuple

import numpy as np

from .causality import time_varying_causality
from .config import PipelineConfig
from .models import compute_multi_trial_BIC
from .simulation import simulate_ar_event_bootstrap
from .utils import (compute_event_statistics, extract_event_snapshots,
                    get_residuals, remove_artifact_trials)
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
            (config.options), detection (config.detection), BIC (config.bic),
            causality (config.causal), bootstrapping (config.monte_carlo),
            and output (config.output). See PipelineConfig definition for details.

    Returns:
        Tuple[Dict, PipelineConfig, np.ndarray]:
            - SnapAnalyOutput (Dict): Dictionary containing analysis results:
                - 'd0' (float | None): Detection threshold used, if calculated.
                - 'locs' (np.ndarray): Detected and filtered event locations (indices).
                - 'morder' (int): The VAR model order used for analysis (potentially BIC selected).
                - 'Yt_stats' (Dict): Statistics computed from event snapshots (mean, covariance, OLS coeffs).
                - 'CausalOutput' (Dict | None): Causality measures (DCS, TE, rDCS) if calculated.
                - 'BICoutputs' (Dict | None): Results from BIC selection, if run.
            - config (PipelineConfig): The input configuration object (potentially updated
              if future versions modify it; currently returns the input config).
            - Yt_events (np.ndarray): Extracted event snapshots used for analysis, shape
              (n_vars * (model_order + 1), L_extract, n_trials_after_artifact_removal).

    Raises:
        ValueError: If configuration parameters are inconsistent (e.g., detection
            is off but no locs provided, align_type invalid). Checked in PipelineConfig.__post_init__.
        TypeError: If input signals are not NumPy arrays or config is not PipelineConfig.
        KeyError: If essential results are unexpectedly missing during internal steps
             (should be less likely with structured config).
        Exception: Can propagate exceptions from underlying functions (e.g., linalg errors).
    """
    # --- Input Validation ---
    # if not isinstance(original_signal, np.ndarray):
    #     logger.warning("original_signal must be a NumPy array.")
    #     original_signal = np.array(original_signal)
    # if not isinstance(detection_signal, np.ndarray):
    #     logger.warning("detection_signal must be a NumPy array.")
    #     detection_signal = np.array(detection_signal)
    # if not isinstance(config, PipelineConfig):
    #     raise TypeError("config must be a PipelineConfig object.")
    # if original_signal.ndim != 2:
    #     raise ValueError("original_signal must be 2D (n_vars, time).")
    # if detection_signal.ndim != 2 or detection_signal.shape[0] != 2:
    #     raise ValueError("detection_signal must be 2D with shape (2, time).")
    # if original_signal.shape[1] != detection_signal.shape[1]:
    #     raise ValueError(
    #         "original_signal and detection_signal must have the same time dimension length."
    #     )

    snap_analysis_output = {}
    logger.info("Starting snapshot detection and analysis pipeline.")

    # --- Step 1: Detect reference points ---
    if config.options.detection:
        logger.info("Performing event detection.")
        D = detection_signal[0]
        d0 = np.nanmean(D) + config.detection.thres_ratio * np.nanstd(D)
        temp_locs = np.where(D >= d0)[0]
        logger.info(
            f"Initial detection: {len(temp_locs)} points above threshold {d0:.2f}."
        )

        align_type = config.detection.align_type
        l_extract = (
            config.detection.l_extract
        )  # Use L_extract for window size in find_peak_loc
        if align_type == "peak":
            locs = find_peak_loc(detection_signal[1], temp_locs, l_extract)
            logger.info(f"Aligned to peaks, found {len(locs)} locations.")
        elif align_type == "pooled":
            if config.detection.shrink_flag:
                pool_window = int(np.ceil(l_extract / 2))
                temp_locs_shrink = shrink_locs_resample_uniform(temp_locs, pool_window)
                locs, _ = find_best_shrinked_locs(D, temp_locs_shrink, temp_locs)
                logger.info(
                    f"Used pooled alignment with shrinking, found {len(locs)} locations."
                )
            else:
                locs = temp_locs  # Use original thresholded if no shrinking
                logger.info(
                    f"Used pooled alignment (no shrinking), using {len(locs)} locations."
                )
    else:
        logger.info("Skipping detection, using provided locations.")
        locs = config.detection.locs
        if locs is None:
            raise ValueError(
                "config.detection.locs cannot be None when config.options.detection is False"
            )
        d0 = None

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
            print(bic_outputs.keys())
            selected_morder = bic_outputs["mobic"][1]
            if np.isnan(selected_morder):
                logger.warning(
                    "BIC calculation resulted in NaN optimal order. Using default morder."
                )
            else:
                morder = int(selected_morder)  # Update morder based on BIC result
                logger.info(f"BIC selected model order: {morder}")
        except Exception as e:
            logger.error(f"BIC calculation failed: {e}. Using default morder: {morder}")
            raise RuntimeError(f"BIC calculation failed: {e}") from e

    # --- Step 4: Extract event snapshots with final morder ---
    # Determine lag step tau: use BIC tau if BIC ran, otherwise default to 1?
    final_tau = config.bic.tau if config.options.bic else 1
    logger.info(
        f"Extracting final event snapshots (morder={morder}, tau={final_tau})..."
    )
    event_snapshots = extract_event_snapshots(
        original_signal, locs, morder, final_tau, config.detection.l_start, l_extract
    )
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
            snap_analysis_output.update({"locs": locs, "morder": morder})
            return (
                snap_analysis_output,
                config,
                event_snapshots,
            )  # Return empty snapshots

    # --- Step 6: Compute statistics ---
    logger.info("Computing event statistics...")
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
            "morder": morder,  # Use the final determined model order
            "diag_flag": config.causal.diag_flag,
            "old_version": config.causal.old_version,
        }
        try:
            # Assuming only OLS mode for now based on dict structure
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
    if config.options.bootstrap:
        logger.info(f"Starting bootstrapping ({config.monte_carlo.n_btsp} samples)...")
        simobj_dict = {
            "nvar": event_stats["OLS"]["At"].shape[1],  # Get nvar from stats
            "morder": morder,
            "L": l_extract,
            "Ntrials": event_snapshots.shape[2],  # Use current number of trials
        }
        try:
            residuals = get_residuals(event_snapshots, event_stats)
            logger.info("Calculated residuals for bootstrapping.")
        except Exception as e:
            logger.error(f"Failed to get residuals for bootstrapping: {e}")
            config.options.bootstrap = False
            logger.warning("Disabling bootstrap due to error in residual calculation.")

        bootstrap_results_list = []

        if config.options.bootstrap:  # Check again in case it was disabled
            for n_btsp in range(1, config.monte_carlo.n_btsp + 1):
                logger.debug(f"Calculating bootstrap trial: {n_btsp}")
                try:
                    btsp_snapshots = simulate_ar_event_bootstrap(
                        simobj_dict, event_snapshots, event_stats, residuals
                    )
                    btsp_stats = compute_event_statistics(btsp_snapshots, morder)
                    btsp_causal_output = {
                        "OLS": time_varying_causality(
                            btsp_snapshots, btsp_stats, causal_params_dict
                        )
                    }
                    bootstrap_results_list.append(btsp_causal_output)

                    if config.options.save_flag:
                        outfile = f"{config.output.file_keyword}_btsp_{n_btsp}_model_causality.npz"
                        np.savez_compressed(
                            outfile,
                            params=config,
                            CausalOutput_btsp=btsp_causal_output,
                            Yt_stats_btsp=btsp_stats,
                        )
                        logger.debug(f"Saved bootstrap sample {n_btsp} to {outfile}")

                except Exception as e:
                    logger.error(f"Bootstrap trial {n_btsp} failed: {e}")

    # --- Step 10: Prepare final output and save ---
    snap_analysis_output.update(
        {
            "d0": d0,
            "locs": locs,
            "morder": morder,
            "Yt_stats": event_stats,  # Contains coefficients, covariances etc.
            "CausalOutput": causal_output,  # Contains DCS, TE, rDCS
            "BICoutputs": bic_outputs,
        }
    )

    if config.options.save_flag:
        file_keyword = config.output.file_keyword
        outfile_final = f"{file_keyword}_model_causality.npz"
        logger.info(f"Saving final results to {outfile_final}...")
        event_stats["mean"] = event_stats["mean"][:2, :]
        event_stats["Sigma"] = event_stats["Sigma"][:, :2, :2]
        try:
            import dataclasses

            config_dict = dataclasses.asdict(config)
            np.savez_compressed(
                outfile_final,
                Config=config_dict,
                SnapAnalyOutput=snap_analysis_output,
                EventSnapshots=event_snapshots,
                Yt_stats=event_stats,
            )
            logger.info("Final results saved.")
        except Exception as e:
            logger.error(f"Failed to save final results: {e}")

    logger.info("Snapshot detection and analysis pipeline finished.")
    return snap_analysis_output, config, event_snapshots
