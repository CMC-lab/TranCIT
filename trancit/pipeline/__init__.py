"""
Pipeline module for Dynamic Causal Strength (DCS) analysis.

This module provides the pipeline orchestration functionality for DCS analysis,
including stage-based processing and the legacy snapshot_detect_analysis_pipeline function.
"""

from .orchestrator import PipelineOrchestrator, PipelineResult
from .stages import (
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

__all__ = [
    # Main pipeline classes
    "PipelineOrchestrator",
    "PipelineResult",
    # Stage classes
    "InputValidationStage",
    "EventDetectionStage",
    "BorderRemovalStage",
    "BICSelectionStage",
    "SnapshotExtractionStage",
    "ArtifactRemovalStage",
    "StatisticsComputationStage",
    "CausalityAnalysisStage",
    "BootstrapAnalysisStage",
    "DeSnapAnalysisStage",
    "OutputPreparationStage",
]
