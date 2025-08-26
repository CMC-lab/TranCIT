"""
Simulation module for the TranCIT package.

This module provides various simulation functions for generating synthetic time series data
for testing and demonstration purposes.

Submodules:
- ar_simulation: Autoregressive process simulations
- oscillator_simulation: Coupled oscillator simulations
- var_simulation: Vector autoregressive simulations
- utils: Simulation utilities (wavelets, etc.)
"""

from .ar_simulation import (
    simulate_ar_event,
    simulate_ar_event_bootstrap,
    simulate_ar_nonstat_innomean,
)
from .oscillator_simulation import generate_signals
from .utils import morlet
from .var_simulation import (
    generate_ensemble_nonstat_innomean,
    generate_var_nonstat,
)

__all__ = [
    "generate_signals",
    "simulate_ar_event",
    "simulate_ar_event_bootstrap",
    "simulate_ar_nonstat_innomean",
    "generate_ensemble_nonstat_innomean",
    "generate_var_nonstat",
    "morlet",
]
