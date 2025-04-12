from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np


@dataclass
class PipelineOptions:
    """Options to control which pipeline steps are executed."""

    detection: bool = True
    bic: bool = False
    causal_analysis: bool = True
    bootstrap: bool = False
    save_flag: bool = False


@dataclass
class DetectionParams:
    """Parameters for the event detection step."""

    thres_ratio: float
    align_type: str  # 'peak' or 'pooled'
    l_extract: int
    l_start: int
    shrink_flag: bool = False
    locs: Optional[np.ndarray] = None  # Provide if Options.detection is False
    remove_artif: bool = False


@dataclass
class BicParams:
    """Parameters for BIC model order selection."""

    morder: int  # Model order to use if BIC is False, or default
    momax: Optional[int] = None  # Max order to test if BIC is True
    tau: Optional[int] = None  # Lag step if BIC is True
    mode: Optional[str] = None  # e.g., 'biased', needed if BIC is True


@dataclass
class CausalParams:
    """Parameters for causality calculation."""

    ref_time: int
    estim_mode: str = "OLS"  # 'OLS' or 'RLS'
    diag_flag: bool = False
    old_version: bool = False


@dataclass
class MonteCParams:
    """Parameters for Monte Carlo bootstrapping."""

    n_btsp: int = 100  # Example default, needed if Options.bootstrap is True


@dataclass
class OutputParams:
    """Parameters for output file naming."""

    file_keyword: str


# --- Main Configuration Class ---
@dataclass
class PipelineConfig:
    """Main configuration object for the analysis pipeline."""

    options: PipelineOptions
    detection: DetectionParams
    bic: BicParams
    causal: CausalParams
    output: OutputParams
    monte_carlo: Optional[MonteCParams] = None
        
    Fs: int = 1252
    passband: List[int] = field(default_factory=lambda: [140, 230])

    # Validation logic
    def __post_init__(self):
        if not self.options.detection and self.detection.locs is None:
            raise ValueError(
                "detection.locs must be provided if options.detection is False"
            )
        if self.options.bic and (
            self.bic.momax is None or self.bic.tau is None or self.bic.mode is None
        ):
            raise ValueError(
                "bic.momax, bic.tau, and bic.mode must be set if options.bic is True"
            )
        if self.options.bootstrap and self.monte_carlo is None:
            raise ValueError(
                "monte_carlo parameters must be provided if options.bootstrap is True"
            )
        if self.detection.align_type not in ["peak", "pooled"]:
            raise ValueError(
                f"Invalid detection.align_type: {self.detection.align_type}. Must be 'peak' or 'pooled'."
            )
