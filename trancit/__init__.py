"""
TranCIT: Transient Causal Interaction Toolbox

A Python package for quantifying causal relationships in multivariate time series data
using dynamic causal strength (DCS) methods.

This package provides methods for analyzing directional influences using model-based
statistical tools, inspired by information-theoretic and autoregressive frameworks.

The package implements several causality measures:
- Dynamic Causal Strength (DCS): Time-varying causal relationships
- Transfer Entropy (TE): Information-theoretic causality measures
- Granger Causality (GC): Linear causality detection
- Relative Dynamic Causal Strength (rDCS): Event-based causality

Example
-------
>>> import numpy as np
>>> from trancit import DCSCalculator
>>>
>>> # Create calculator
>>> calculator = DCSCalculator(model_order=4, time_mode="inhomo")
>>>
>>> # Generate sample data
>>> data = np.random.randn(2, 1000, 20)  # (n_vars, n_obs, n_trials)
>>>
>>> # Perform analysis
>>> result = calculator.analyze(data)
>>> print(f"DCS shape: {result.causal_strength.shape}")
"""

# Version is automatically set by setuptools_scm from git tags
try:
    from ._version import version as __version__
except ImportError:
    # Fallback for development installs without setuptools_scm
    __version__ = "0.1.0-dev"
__author__ = "Salar Nouri"
__email__ = "salr.nouri@gmail.com"
__license__ = "BSD-2-Clause"

# Causality analysis imports
from trancit.causality.dcs import DCSCalculator, DCSResult
from trancit.causality.granger import GrangerCausalityCalculator, GrangerCausalityResult
from trancit.causality.rdcs import (
    RelativeDCSCalculator,
    RelativeDCSResult,
    time_varying_causality,
)
from trancit.causality.transfer_entropy import (
    TransferEntropyCalculator,
    TransferEntropyResult,
)

# Configuration imports
from trancit.config import (
    BicParams,
    CausalParams,
    DeSnapParams,
    DetectionParams,
    MonteCParams,
    OutputParams,
    PipelineConfig,
    PipelineOptions,
)

# Core functionality imports
from trancit.core.base import BaseAnalyzer, BaseConfig, BaseResult
from trancit.core.exceptions import (
    ComputationError,
    ConfigurationError,
    ConvergenceError,
    DataError,
    DCSError,
    SingularMatrixError,
    ValidationError,
)

# Setup logging
from trancit.logger_config import setup_logging
from trancit.models.bic_selection import BICSelector
from trancit.models.model_validation import ModelValidator

# Model estimation imports
from trancit.models.var_estimation import VAREstimator

# Pipeline imports
from trancit.pipeline.orchestrator import PipelineOrchestrator, PipelineResult
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

# Simulation imports
from trancit.simulation import (
    generate_ensemble_nonstat_innomean,
    generate_signals,
    generate_var_nonstat,
    morlet,
    simulate_ar_event,
    simulate_ar_event_bootstrap,
    simulate_ar_nonstat_innomean,
)

# Utility imports
from trancit.utils import (
    compute_event_statistics,
    estimate_residuals,
    extract_event_snapshots,
    find_best_shrinked_locations,
    find_peak_locations,
    get_residuals,
    perform_desnap_analysis,
    remove_artifact_trials,
    shrink_locations_resample_uniform,
)

setup_logging(log_file="trancit_log.txt")

# Public API definition
__all__ = [
    # Core classes
    "BaseAnalyzer",
    "BaseResult",
    "BaseConfig",
    # Exceptions
    "DCSError",
    "ValidationError",
    "ComputationError",
    "ConfigurationError",
    "DataError",
    "ConvergenceError",
    "SingularMatrixError",
    # Configuration classes
    "PipelineConfig",
    "PipelineOptions",
    "DetectionParams",
    "BicParams",
    "CausalParams",
    "MonteCParams",
    "OutputParams",
    "DeSnapParams",
    # Causality analysis
    "DCSCalculator",
    "DCSResult",
    "TransferEntropyCalculator",
    "TransferEntropyResult",
    "GrangerCausalityCalculator",
    "GrangerCausalityResult",
    "RelativeDCSCalculator",
    "RelativeDCSResult",
    "time_varying_causality",
    # Model estimation
    "VAREstimator",
    "BICSelector",
    "ModelValidator",
    # Pipeline stages
    "PipelineOrchestrator",
    "PipelineResult",
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
    # Simulation functions
    "generate_signals",
    "simulate_ar_event",
    "simulate_ar_event_bootstrap",
    "simulate_ar_nonstat_innomean",
    "generate_ensemble_nonstat_innomean",
    "generate_var_nonstat",
    "morlet",
    # Utility functions
    "compute_event_statistics",
    "perform_desnap_analysis",
    "estimate_residuals",
    "extract_event_snapshots",
    "get_residuals",
    "remove_artifact_trials",
    "find_peak_locations",
    "shrink_locations_resample_uniform",
    "find_best_shrinked_locations",
]
