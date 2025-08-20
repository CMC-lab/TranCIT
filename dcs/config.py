"""
Configuration classes for the Dynamic Causal Strength (DCS) package.

This module provides dataclasses for configuring the analysis pipeline,
including detection parameters, BIC model selection, causality analysis,
and output settings.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np


@dataclass
class PipelineOptions:
    """
    Options to control which pipeline steps are executed.

    Attributes
    ----------
    detection : bool
        Whether to perform event detection (default: True)
    bic : bool
        Whether to perform BIC model selection (default: False)
    causal_analysis : bool
        Whether to perform causality analysis (default: True)
    bootstrap : bool
        Whether to perform bootstrap analysis (default: False)
    save_flag : bool
        Whether to save results to files (default: False)
    debiased_stats : bool
        Whether to perform DeSnap analysis (default: True)
    """

    detection: bool = True
    bic: bool = False
    causal_analysis: bool = True
    bootstrap: bool = False
    save_flag: bool = False
    debiased_stats: bool = True


@dataclass
class DetectionParams:
    """
    Parameters for the event detection step.

    Attributes
    ----------
    thres_ratio : float
        Threshold ratio for event detection
    align_type : str
        Alignment type: 'peak' or 'pooled'
    l_extract : int
        Length of extracted snapshots
    l_start : int
        Start offset for snapshot extraction
    shrink_flag : bool
        Whether to use shrinking for pooled alignment (default: False)
    locs : Optional[np.ndarray]
        Pre-provided event locations (default: None)
    remove_artif : bool
        Whether to remove artifact trials (default: False)
    """

    thres_ratio: float
    align_type: str  # 'peak' or 'pooled'
    l_extract: int
    l_start: int
    shrink_flag: bool = False
    locs: Optional[np.ndarray] = None
    remove_artif: bool = False
    remove_artif_threshold: float = -15000


@dataclass
class BicParams:
    """
    Parameters for BIC model order selection.

    Attributes
    ----------
    morder : int
        Model order to use if BIC is False, or default
    momax : Optional[int]
        Max order to test if BIC is True (default: None)
    tau : Optional[int]
        Lag step if BIC is True (default: None)
    mode : Optional[str]
        BIC mode, e.g., 'biased' (default: None)
    """

    morder: int
    momax: Optional[int] = None
    tau: Optional[int] = None
    mode: Optional[str] = None
    estim_mode: Optional[str] = None


@dataclass
class CausalParams:
    """
    Parameters for causality calculation.

    Attributes
    ----------
    ref_time : int
        Reference time point for causality analysis
    estim_mode : str
        Estimation mode: 'OLS' or 'RLS' (default: 'OLS')
    diag_flag : bool
        Whether to use diagonal covariance (default: False)
    old_version : bool
        Whether to use old version of rDCS calculation (default: False)
    """

    ref_time: int
    estim_mode: str = "OLS"  # 'OLS' or 'RLS'
    diag_flag: bool = False
    old_version: bool = False


@dataclass
class MonteCParams:
    """
    Parameters for Monte Carlo bootstrapping.

    Attributes
    ----------
    n_btsp : int
        Number of bootstrap samples (default: 100)
    """

    n_btsp: int = 100


@dataclass
class OutputParams:
    """
    Parameters for output file naming.

    Attributes
    ----------
    file_keyword : str
        Keyword for output file naming
    save_path : str
        Path for saving output files
    """

    file_keyword: str
    save_path: str = ""


@dataclass
class DeSnapParams:
    """
    Input structure for the de-snapshotting analysis.

    Attributes
    ----------
    detection_signal : np.ndarray
        Conditioning variable values (1D array)
    original_signal : np.ndarray
        Original time series data
    Yt_stats_cond : Dict
        Conditional statistics from pipeline
    morder : int
        Model order
    tau : int
        Lag step
    l_start : int
        Start offset for snapshot extraction
    l_extract : int
        Length of extracted snapshots
    d0 : float
        Lower bound of the first bin for D
    N_d : int
        Number of bins for D
    d0_max : Optional[float]
        Upper bound for binning D (default: None)
    maxStdRatio : Optional[float]
        Alternative to define d0_max based on std(D) (default: None)
    diff_flag : bool
        Flag for how 'c' (covariance adjustment factor) is calculated (default: False)
    """

    detection_signal: np.ndarray
    original_signal: np.ndarray
    event_stats: Dict
    morder: int
    tau: int
    l_start: int
    l_extract: int
    d0: float
    N_d: int
    d0_max: Optional[float] = None
    maxStdRatio: Optional[float] = None
    diff_flag: bool = False


@dataclass
class PipelineConfig:
    """
    Main configuration object for the analysis pipeline.

    This class combines all configuration parameters into a single object
    for easy management and validation of pipeline settings.

    Attributes
    ----------
    options : PipelineOptions
        Pipeline execution options
    detection : DetectionParams
        Event detection parameters
    bic : BicParams
        BIC model selection parameters
    causal : CausalParams
        Causality analysis parameters
    output : OutputParams
        Output file parameters
    monte_carlo : Optional[MonteCParams]
        Monte Carlo bootstrap parameters (default: None)
    desnap : Optional[DeSnapParams]
        DeSnap analysis parameters (default: None)
    sampling_rate : int
        Sampling rate in Hz (default: 1252)
    passband : List[int]
        Frequency passband [low, high] in Hz (default: [140, 230])
    """

    options: PipelineOptions
    detection: DetectionParams
    bic: BicParams
    causal: CausalParams
    output: OutputParams
    monte_carlo: Optional[MonteCParams] = None
    desnap: Optional[DeSnapParams] = None
    sampling_rate: int = 1252
    passband: List[int] = field(default_factory=lambda: [140, 230])

    def __post_init__(self) -> None:
        """Validate configuration parameters after initialization."""
        self._validate_detection_params()
        self._validate_bic_params()
        self._validate_bootstrap_params()
        self._validate_align_type()

    def _validate_detection_params(self) -> None:
        """Validate detection parameters."""
        if not self.options.detection and self.detection.locs is None:
            raise ValueError(
                "detection.locs must be provided if options.detection is False"
            )

    def _validate_bic_params(self) -> None:
        """Validate BIC parameters."""
        if self.options.bic and (
            self.bic.momax is None or self.bic.tau is None or self.bic.mode is None
        ):
            raise ValueError(
                "bic.momax, bic.tau, and bic.mode must be set if options.bic is True"
            )

    def _validate_bootstrap_params(self) -> None:
        """Validate bootstrap parameters."""
        if self.options.bootstrap and self.monte_carlo is None:
            raise ValueError(
                "monte_carlo parameters must be provided if options.bootstrap is True"
            )

    def _validate_align_type(self) -> None:
        """Validate alignment type."""
        if self.detection.align_type not in ["peak", "pooled"]:
            raise ValueError(
                f"Invalid detection.align_type: {self.detection.align_type}. "
                "Must be 'peak' or 'pooled'."
            )
