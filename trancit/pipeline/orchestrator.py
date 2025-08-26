"""
Pipeline orchestrator for Dynamic Causal Strength (DCS) analysis.

This module provides the main pipeline orchestrator that coordinates
all stages of the DCS analysis pipeline.
"""

import logging
from typing import Any, Dict

import numpy as np

from trancit.config import PipelineConfig
from trancit.core.base import BaseAnalyzer, BaseResult
from trancit.core.exceptions import ComputationError, ValidationError
from trancit.pipeline.stages import (
    ArtifactRemovalStage,
    BICSelectionStage,
    BootstrapAnalysisStage,
    BorderRemovalStage,
    CausalityAnalysisStage,
    DeSnapAnalysisStage,
    EventDetectionStage,
    InputValidationStage,
    OutputPreparationStage,
    SnapshotExtractionStage,
    StatisticsComputationStage,
)

logger = logging.getLogger(__name__)


class PipelineResult(BaseResult):
    """Result container for pipeline analysis."""

    def __init__(
        self,
        results: Dict[str, Any],
        config: PipelineConfig,
        event_snapshots: np.ndarray,
    ):
        """
        Initialize pipeline result.

        Parameters
        ----------
        results : Dict[str, Any]
            Analysis results
        config : PipelineConfig
            Pipeline configuration
        event_snapshots : np.ndarray
            Extracted event snapshots
        """
        super().__init__(
            results=results,
            config=config,
            event_snapshots=event_snapshots,
        )


class PipelineOrchestrator(BaseAnalyzer):
    """
    Pipeline orchestrator for DCS analysis.

    This class coordinates all stages of the DCS analysis pipeline,
    including event detection, snapshot extraction, and causality analysis.
    """

    def __init__(self, config: PipelineConfig):
        """
        Initialize pipeline orchestrator.

        Parameters
        ----------
        config : PipelineConfig
            Pipeline configuration
        """
        super().__init__(config=config)
        self.config = config
        self._initialize_stages()

    def _initialize_stages(self) -> None:
        """Initialize pipeline stages."""
        self.stages = {
            "input_validation": InputValidationStage(self.config),
            "event_detection": EventDetectionStage(self.config),
            "border_removal": BorderRemovalStage(self.config),
            "bic_selection": BICSelectionStage(self.config),
            "snapshot_extraction": SnapshotExtractionStage(self.config),
            "artifact_removal": ArtifactRemovalStage(self.config),
            "statistics_computation": StatisticsComputationStage(self.config),
            "causality_analysis": CausalityAnalysisStage(self.config),
            "bootstrap_analysis": BootstrapAnalysisStage(self.config),
            "desnap_analysis": DeSnapAnalysisStage(self.config),
            "output_preparation": OutputPreparationStage(self.config),
        }

    def analyze(self, data: np.ndarray, **kwargs) -> PipelineResult:
        """
        Perform pipeline analysis on the given data.

        This method implements the BaseAnalyzer interface and serves as a wrapper
        around the run method for compatibility.

        Parameters
        ----------
        data : np.ndarray
            Original time series signal (first argument)
        **kwargs
            Additional parameters including:
            - detection_signal: Signal used for event detection

        Returns
        -------
        PipelineResult
            Pipeline analysis results
        """
        if "detection_signal" not in kwargs:
            raise ValueError("detection_signal must be provided in kwargs")

        return self.run(data, kwargs["detection_signal"])

    def run(
        self,
        original_signal: np.ndarray,
        detection_signal: np.ndarray,
    ) -> PipelineResult:
        """
        Run the complete DCS analysis pipeline.

        Parameters
        ----------
        original_signal : np.ndarray
            Original time series signal
        detection_signal : np.ndarray
            Signal used for event detection

        Returns
        -------
        PipelineResult
            Pipeline analysis results
        """
        self._validate_input_data(original_signal)
        self._validate_input_data(detection_signal)
        self._log_analysis_start((original_signal.shape, detection_signal.shape))

        try:
            pipeline_state = {
                "original_signal": original_signal,
                "detection_signal": detection_signal,
            }

            pipeline_state = self._execute_stage("input_validation", pipeline_state)
            pipeline_state = self._execute_stage("event_detection", pipeline_state)
            pipeline_state = self._execute_stage("border_removal", pipeline_state)
            pipeline_state = self._execute_stage("bic_selection", pipeline_state)
            pipeline_state = self._execute_stage("snapshot_extraction", pipeline_state)
            pipeline_state = self._execute_stage("artifact_removal", pipeline_state)
            pipeline_state = self._execute_stage(
                "statistics_computation", pipeline_state
            )

            if self.config.options.causal_analysis:
                pipeline_state = self._execute_stage(
                    "causality_analysis", pipeline_state
                )

            if self.config.options.bootstrap:
                pipeline_state = self._execute_stage(
                    "bootstrap_analysis", pipeline_state
                )

            if self.config.options.debiased_stats:
                pipeline_state = self._execute_stage("desnap_analysis", pipeline_state)

            # pipeline_state = self._execute_stage("output_preparation", pipeline_state)

            self._ensure_final_results(
                pipeline_state, original_signal, detection_signal
            )

            self._log_analysis_complete()

            return PipelineResult(
                results=pipeline_state["final_results"],
                config=self.config,
                event_snapshots=pipeline_state.get("event_snapshots", np.array([])),
            )

        except Exception as e:
            logger.error(f"Pipeline execution failed: {e}")
            raise ComputationError(
                f"Pipeline execution failed: {e}",
                "pipeline_execution",
                original_signal.shape,
            )

    def _execute_stage(
        self, stage_name: str, pipeline_state: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Execute a single pipeline stage."""
        try:
            stage = self.stages[stage_name]
            logger.info(f"Executing stage: {stage_name}")

            stage_result = stage.execute(**pipeline_state)
            pipeline_state.update(stage_result)

            logger.info(f"Completed stage: {stage_name}")
            return pipeline_state

        except Exception as e:
            logger.error(f"Stage {stage_name} failed: {e}")
            raise ComputationError(f"Stage {stage_name} failed: {e}", stage_name, ())

    def _validate_input_data(self, data: np.ndarray) -> None:
        """Validate input data format and dimensions."""
        super()._validate_input_data(data)

        if data.ndim != 2:
            raise ValidationError(
                "Input signals must be 2D (n_vars, time)", "data_ndim", data.ndim
            )

        if data.shape[0] != 2:
            raise ValidationError(
                "Input signals must be bivariate (n_vars=2)", "n_vars", data.shape[0]
            )

    def _ensure_final_results(
        self,
        pipeline_state: Dict[str, Any],
        original_signal: np.ndarray,
        detection_signal: np.ndarray,
    ) -> None:
        """Ensure final results are present in the pipeline state."""
        if "final_results" not in pipeline_state:
            pipeline_state["final_results"] = {
                "status": "completed",
                "message": "Pipeline executed with minimal stages",
                "original_signal_shape": original_signal.shape,
                "detection_signal_shape": detection_signal.shape,
            }
