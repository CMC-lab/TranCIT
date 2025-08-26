"""
Causality analysis module for TranCIT: Transient Causal Interaction.

This module provides implementations of various causality measures:
- Dynamic Causal Strength (DCS): Time-varying causal relationships
- Transfer Entropy (TE): Information-theoretic causality measures
- Granger Causality (GC): Linear causality detection
- Relative Dynamic Causal Strength (rDCS): Event-based causality

All causality calculators inherit from BaseCausalityAnalyzer and provide
a consistent interface for analyzing causal relationships in time series data.

Example
-------
>>> import numpy as np
>>> from trancit.causality import DCSCalculator
>>>
>>> # Create DCS calculator
>>> calculator = DCSCalculator(model_order=4, time_mode="inhomo")
>>>
>>> # Analyze data
>>> data = np.random.randn(2, 1000, 20)  # (n_vars, n_obs, n_trials)
>>> result = calculator.analyze(data)
>>>
>>> # Access results
>>> print(f"DCS: {result.causal_strength.shape}")
>>> print(f"TE: {result.transfer_entropy.shape}")
>>> print(f"GC: {result.granger_causality.shape}")
"""

from .base import BaseCausalityAnalyzer, CausalityResult
from .dcs import DCSCalculator, DCSResult
from .granger import GrangerCausalityCalculator, GrangerCausalityResult
from .rdcs import RelativeDCSCalculator, RelativeDCSResult, time_varying_causality
from .transfer_entropy import TransferEntropyCalculator, TransferEntropyResult

__all__ = [
    "CausalityResult",
    "BaseCausalityAnalyzer",
    "DCSCalculator",
    "DCSResult",
    "TransferEntropyCalculator",
    "TransferEntropyResult",
    "GrangerCausalityCalculator",
    "GrangerCausalityResult",
    "RelativeDCSCalculator",
    "RelativeDCSResult",
    "time_varying_causality",
]
