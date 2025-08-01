"""
Simulation module for the DCS package.

This module provides various simulation functions for generating synthetic time series data
for testing and demonstration purposes.

Submodules:
- ar_simulation: Autoregressive process simulations
- oscillator_simulation: Coupled oscillator simulations  
- var_simulation: Vector autoregressive simulations
- utils: Simulation utilities (wavelets, etc.)
"""

from .oscillator_simulation import generate_signals
from .ar_simulation import (
    simulate_ar_event,
    simulate_ar_event_bootstrap,
    simulate_ar_nonstat_innomean,
)
from .var_simulation import (
    generate_ensemble_nonstat_innomean,
    generate_var_nonstat,
)
from .utils import morlet

__all__ = [
    "generate_signals",
    "simulate_ar_event", 
    "simulate_ar_event_bootstrap",
    "simulate_ar_nonstat_innomean",
    "generate_ensemble_nonstat_innomean",
    "generate_var_nonstat",
    "morlet",
] 
