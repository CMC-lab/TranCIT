"""
Pipeline stages for Dynamic Causal Strength (DCS) analysis.

This module provides individual pipeline stages that can be composed
to create the complete analysis pipeline.
"""

import logging
from abc import ABC, abstractmethod
from typing import Any, Dict

import numpy as np

from trancit.config import PipelineConfig
from trancit.utils import (
    compute_event_statistics,
    extract_event_snapshots,
    remove_artifact_trials,
)
from trancit.utils.signal import (
    find_best_shrinked_locations,
    find_peak_locations,
    shrink_locations_resample_uniform,
)

logger = logging.getLogger(__name__)


class PipelineStage(ABC):
    """Abstract base class for pipeline stages."""

    def __init__(self, config: PipelineConfig):
        self.config = config

    @abstractmethod
    def execute(self, **kwargs) -> Dict[str, Any]:
        """Execute the pipeline stage."""

    def _log_stage_start(self, stage_name: str) -> None:
        """Log the start of a pipeline stage."""
        logger.info(f"Starting {stage_name} stage")

    def _log_stage_complete(self, stage_name: str) -> None:
        """Log the completion of a pipeline stage."""
        logger.info(f"Completed {stage_name} stage")


class InputValidationStage(PipelineStage):
    """Stage for validating input parameters and data."""

    def execute(self, **kwargs) -> Dict[str, Any]:
        """Validate input parameters and data."""
        self._log_stage_start("input validation")

        original_signal = kwargs.get("original_signal")
        detection_signal = kwargs.get("detection_signal")

        if not isinstance(original_signal, np.ndarray):
            raise TypeError("original_signal must be a NumPy array")

        if not isinstance(detection_signal, np.ndarray):
            raise TypeError("detection_signal must be a NumPy array")

        if original_signal.ndim != 2:
            raise ValueError("original_signal must be 2D (n_vars, time)")

        if detection_signal.ndim != 2 or detection_signal.shape[0] != 2:
            raise ValueError("detection_signal must be 2D with shape (2, time)")

        if original_signal.shape[1] != detection_signal.shape[1]:
            logger.warning(
                "original_signal and detection_signal must have the same time dimension"
            )

        self._log_stage_complete("input validation")
        return {
            "original_signal": original_signal,
            "detection_signal": detection_signal,
        }


class EventDetectionStage(PipelineStage):
    """Stage for detecting and aligning events."""

    def execute(self, **kwargs) -> Dict[str, Any]:
        """Detect and align events based on the detection signal."""
        self._log_stage_start("event detection")

        detection_signal = kwargs["detection_signal"]
        D_for_detection = detection_signal[0]
        d0_threshold = None

        if self.config.options.detection:
            d0_threshold = np.nanmean(
                D_for_detection
            ) + self.config.detection.thres_ratio * np.nanstd(D_for_detection)
            temp_locs = np.where(D_for_detection >= d0_threshold)[0]
            logger.info(
                f"Initial detection: {len(temp_locs)} points above threshold {d0_threshold:.2f}"
            )

            align_type = self.config.detection.align_type
            if align_type == "peak":
                locs = find_peak_locations(
                    detection_signal[1], temp_locs, self.config.detection.l_extract
                )
                logger.info(f"Aligned to peaks, found {len(locs)} locations.")
            elif align_type == "pooled":
                if self.config.detection.shrink_flag:
                    pool_window = int(np.ceil(self.config.detection.l_extract / 2))
                    temp_locs_shrink = shrink_locations_resample_uniform(
                        temp_locs, pool_window
                    )
                    locs, _ = find_best_shrinked_locations(
                        D_for_detection, temp_locs_shrink, temp_locs
                    )
                    logger.info(
                        f"Used pooled alignment with shrinking, found {len(locs)} locations."
                    )
                else:
                    locs = temp_locs
                    logger.info(
                        f"Used pooled alignment (no shrinking), using {len(locs)} locations."
                    )
        else:
            if self.config.detection.locs is None:
                raise ValueError(
                    "config.detection.locs cannot be None when config.options.detection is False"
                )
            locs = np.array(self.config.detection.locs, dtype=int)

        self._log_stage_complete("event detection")
        return {"locs": locs, "d0_threshold": d0_threshold}


class BorderRemovalStage(PipelineStage):
    """Stage for removing event locations too close to signal borders."""

    def execute(self, **kwargs) -> Dict[str, Any]:
        """Remove event locations that are too close to signal borders."""
        self._log_stage_start("border removal")

        locs = kwargs["locs"]
        original_signal = kwargs["original_signal"]
        l_extract = self.config.detection.l_extract
        signal_length = original_signal.shape[1]

        original_length = len(locs)
        locs = locs[(locs >= l_extract) & (locs <= signal_length - l_extract)]

        if len(locs) < original_length:
            logger.info(
                f"Removed {original_length - len(locs)} locations too close to signal borders."
            )

        self._log_stage_complete("border removal")
        return {"locs": locs}


class BICSelectionStage(PipelineStage):
    """Stage for BIC model selection."""

    def execute(self, **kwargs) -> Dict[str, Any]:
        """Perform BIC model selection if enabled."""
        self._log_stage_start("BIC selection")

        original_signal = kwargs["original_signal"]
        locs = kwargs["locs"]
        bic_outputs = None
        morder = self.config.bic.morder

        if self.config.options.bic:
            logger.info("Performing BIC model selection.")

            from trancit.models.bic_selection import BICSelector

            try:
                event_snapshots_momax = extract_event_snapshots(
                    original_signal,
                    locs,
                    self.config.bic.momax,
                    self.config.bic.tau,
                    self.config.detection.l_start,
                    self.config.detection.l_extract,
                )

                bic_selector = BICSelector(self.config)
                bic_outputs = bic_selector._compute_multi_trial_bic(
                    event_snapshots_momax
                )

                if (
                    "mobic" in bic_outputs
                    and bic_outputs["mobic"] is not None
                    and len(bic_outputs["mobic"]) > 1
                ):
                    selected_morder = bic_outputs["mobic"][1]
                    if not np.isnan(selected_morder):
                        morder = int(selected_morder)
                        logger.info(f"BIC selected model order: {morder}")
                    else:
                        logger.warning(
                            "BIC calculation resulted in NaN optimal order. Using default morder."
                        )
                else:
                    logger.error(
                        f"Could not find 'mobic' in BIC output: {bic_outputs.keys()}"
                    )
                    raise KeyError("Optimal model order key not found in BIC results.")

            except Exception as e:
                logger.error(
                    f"BIC calculation failed: {e}. Using default morder: {morder}"
                )
                raise RuntimeError(f"BIC calculation failed: {e}") from e
        else:
            logger.info("Skipping BIC model selection.")
            bic_outputs = None

        self.config.bic.morder = morder
        self._log_stage_complete("BIC selection")
        return {"bic_outputs": bic_outputs, "morder": morder}


class SnapshotExtractionStage(PipelineStage):
    """Stage for extracting event snapshots."""

    def execute(self, **kwargs) -> Dict[str, Any]:
        """Extract event snapshots using configuration parameters."""
        self._log_stage_start("snapshot extraction")

        original_signal = kwargs["original_signal"]
        locs = kwargs["locs"]
        morder = kwargs["morder"]

        final_tau = self.config.bic.tau if self.config.options.bic else 1
        logger.info(
            f"Extracting final event snapshots (morder={morder}, tau={final_tau})..."
        )

        try:
            event_snapshots = extract_event_snapshots(
                original_signal,
                locs,
                morder,
                final_tau,
                self.config.detection.l_start,
                self.config.detection.l_extract,
            )
            logger.info(
                f"Extracted event snapshots with shape: {event_snapshots.shape}"
            )

            if event_snapshots.shape[2] == 0:
                logger.warning(
                    "No trials available for final analysis after snapshot extraction."
                )
                return {"event_snapshots": event_snapshots}

        except Exception as e:
            logger.error(f"Failed to extract event snapshots: {e}")
            raise

        self._log_stage_complete("snapshot extraction")
        return {"event_snapshots": event_snapshots}


class ArtifactRemovalStage(PipelineStage):
    """Stage for removing artifact-contaminated trials."""

    def execute(self, **kwargs) -> Dict[str, Any]:
        """Remove artifact-contaminated trials."""
        self._log_stage_start("artifact removal")

        event_snapshots = kwargs["event_snapshots"]
        locs = kwargs["locs"]

        if self.config.detection.remove_artif:
            try:
                threshold = self.config.detection.remove_artif_threshold
                logger.info(f"Removing artifact trials below threshold {threshold}...")

                cleaned_snapshots, cleaned_locs = remove_artifact_trials(
                    event_snapshots, locs, threshold
                )

                removed_count = event_snapshots.shape[2] - cleaned_snapshots.shape[2]
                if removed_count > 0:
                    logger.info(
                        f"Removed {removed_count} artifact trials. "
                        f"{cleaned_snapshots.shape[2]} trials remaining."
                    )
                    event_snapshots = cleaned_snapshots
                    locs = cleaned_locs
                else:
                    logger.info("No artifact trials removed.")

            except Exception as e:
                logger.warning(f"Artifact removal failed: {e}. Using original data.")
        else:
            logger.info("Skipping artifact removal.")

        if event_snapshots.shape[2] == 0:
            logger.warning("No trials remaining after artifact removal.")

        self._log_stage_complete("artifact removal")
        return {"event_snapshots": event_snapshots, "locs": locs}


class StatisticsComputationStage(PipelineStage):
    """Stage for computing event statistics."""

    def execute(self, **kwargs) -> Dict[str, Any]:
        """Compute event statistics with error handling."""
        self._log_stage_start("statistics computation")

        event_snapshots = kwargs["event_snapshots"]
        morder = kwargs["morder"]

        try:
            event_stats = compute_event_statistics(event_snapshots, morder)
            logger.info("Event statistics computed successfully.")
        except Exception as e:
            logger.error(f"Failed to compute event statistics: {e}")
            raise

        self._log_stage_complete("statistics computation")
        return {"event_stats": event_stats}


class CausalityAnalysisStage(PipelineStage):
    """Stage for performing causality analysis."""

    def execute(self, **kwargs) -> Dict[str, Any]:
        """Perform causality analysis if enabled."""
        self._log_stage_start("causality analysis")

        event_snapshots = kwargs["event_snapshots"]
        event_stats = kwargs["event_stats"]

        if not self.config.options.causal_analysis:
            logger.info("Skipping causality analysis.")
            return {"causal_output": None}

        logger.info("Performing causality analysis.")

        from trancit.causality.rdcs import time_varying_causality

        causal_params = {
            "ref_time": self.config.causal.ref_time,
            "estim_mode": self.config.causal.estim_mode,
            "morder": kwargs["morder"],
            "diag_flag": self.config.causal.diag_flag,
            "old_version": self.config.causal.old_version,
        }

        try:
            causal_output = {
                "OLS": time_varying_causality(
                    event_snapshots, event_stats, causal_params
                )
            }
            logger.info("Causality analysis completed successfully.")
        except Exception as e:
            logger.error(f"Causality analysis failed: {e}")
            causal_output = None

        self._log_stage_complete("causality analysis")
        return {"causal_output": causal_output}


class BootstrapAnalysisStage(PipelineStage):
    """Stage for performing bootstrap analysis."""

    def execute(self, **kwargs) -> Dict[str, Any]:
        """Perform bootstrap analysis if enabled."""
        self._log_stage_start("bootstrap analysis")

        if not self.config.options.bootstrap or self.config.monte_carlo is None:
            logger.info("Skipping bootstrap analysis.")
            return {"bootstrap_causal_outputs_list": None}

        logger.info("Performing bootstrap analysis.")

        from trancit.causality.rdcs import time_varying_causality
        from trancit.simulation import simulate_ar_event_bootstrap
        from trancit.utils.residuals import get_residuals

        bootstrap_causal_outputs_list = []

        try:
            event_snapshots = kwargs["event_snapshots"]
            event_stats = kwargs["event_stats"]
            residuals_for_btsp = get_residuals(event_snapshots, event_stats)

            simobj_dict_bootstrap = {
                "nvar": event_stats["OLS"]["At"].shape[1],
                "morder": kwargs["morder"],
                "L": self.config.detection.l_extract,
                "Ntrials": event_snapshots.shape[2],
            }

            causal_params = {
                "ref_time": self.config.causal.ref_time,
                "estim_mode": self.config.causal.estim_mode,
                "morder": kwargs["morder"],
                "diag_flag": self.config.causal.diag_flag,
                "old_version": self.config.causal.old_version,
            }

            for n_btsp in range(1, self.config.monte_carlo.n_btsp + 1):
                logger.info(
                    f"Bootstrap trial {n_btsp} of {self.config.monte_carlo.n_btsp}"
                )
                try:
                    btsp_snapshots = simulate_ar_event_bootstrap(
                        simobj_dict_bootstrap,
                        event_snapshots,
                        event_stats,
                        residuals_for_btsp,
                    )
                    btsp_stats = compute_event_statistics(
                        btsp_snapshots, kwargs["morder"]
                    )

                    btsp_causal_output = {
                        "OLS": time_varying_causality(
                            btsp_snapshots, btsp_stats, causal_params
                        )
                    }
                    bootstrap_causal_outputs_list.append(btsp_causal_output)

                except Exception as e:
                    logger.warning(f"Bootstrap trial {n_btsp} failed: {e}")

        except Exception as e:
            logger.error(f"Bootstrap analysis failed: {e}")
            bootstrap_causal_outputs_list = []

        self._log_stage_complete("bootstrap analysis")
        return {"bootstrap_causal_outputs_list": bootstrap_causal_outputs_list}


class DeSnapAnalysisStage(PipelineStage):
    """Stage for performing DeSnap analysis."""

    def execute(self, **kwargs) -> Dict[str, Any]:
        """Perform DeSnap analysis if enabled."""
        self._log_stage_start("DeSnap analysis")

        if not self.config.options.debiased_stats or self.config.desnap is None:
            logger.info("Skipping DeSnap analysis.")
            return {"desnap_full_output": None, "event_stats_unconditional": None}

        logger.info("Performing DeSnap analysis.")

        from trancit.utils.core import perform_desnap_analysis
        from trancit.utils.residuals import estimate_residuals

        try:
            desnap_params_instance = self.config.desnap
            desnap_params_instance.detection_signal = kwargs["detection_signal"][0]
            desnap_params_instance.original_signal = kwargs["original_signal"]
            desnap_params_instance.event_stats = kwargs["event_stats"]
            desnap_params_instance.tau = self.config.bic.tau
            desnap_params_instance.l_start = self.config.detection.l_start
            desnap_params_instance.l_extract = self.config.detection.l_extract
            desnap_params_instance.morder = kwargs["morder"]
            desnap_params_instance.d0 = kwargs.get("d0_threshold", 0.0)

            desnap_full_output = perform_desnap_analysis(desnap_params_instance)

            event_stats_unconditional = None
            if "Yt_stats_uncond" in desnap_full_output:
                event_stats_unconditional = desnap_full_output["Yt_stats_uncond"]

                if "OLS" not in event_stats_unconditional:
                    event_stats_unconditional["OLS"] = {}

                bt_uncond, sigma_et_uncond, _ = estimate_residuals(
                    event_stats_unconditional
                )
                event_stats_unconditional["OLS"]["bt"] = bt_uncond
                event_stats_unconditional["OLS"]["Sigma_Et"] = sigma_et_uncond
                logger.info("DeSnap analysis complete. Unconditional stats derived.")

        except Exception as e:
            logger.error(f"DeSnap analysis step failed: {e}")
            desnap_full_output = None
            event_stats_unconditional = None

        self._log_stage_complete("DeSnap analysis")
        return {
            "desnap_full_output": desnap_full_output,
            "event_stats_unconditional": event_stats_unconditional,
        }


class OutputPreparationStage(PipelineStage):
    """Stage for preparing final output."""

    def execute(self, **kwargs) -> Dict[str, Any]:
        """Prepare the final output dictionary."""
        self._log_stage_start("output preparation")

        final_results = {
            "d0": kwargs.get("d0_threshold"),
            "locs": kwargs["locs"],
            "morder": kwargs["morder"],
            "Yt_stats": kwargs["event_stats"],
        }

        if kwargs.get("causal_output") is not None:
            final_results["CausalOutput"] = kwargs["causal_output"]

        if kwargs.get("bic_outputs") is not None:
            final_results["BICoutputs"] = kwargs["bic_outputs"]

        if kwargs.get("bootstrap_causal_outputs_list") is not None:
            final_results["BootstrapCausalOutputs"] = kwargs[
                "bootstrap_causal_outputs_list"
            ]

        if kwargs.get("desnap_full_output") is not None:
            final_results["DeSnap_output"] = kwargs["desnap_full_output"]

        if kwargs.get("event_stats_unconditional") is not None:
            final_results["Yt_stats_unconditional"] = kwargs[
                "event_stats_unconditional"
            ]

        self._log_stage_complete("output preparation")
        return {"final_results": final_results}
