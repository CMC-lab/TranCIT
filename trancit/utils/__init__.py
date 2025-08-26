"""
Utilities Sub-package for TranCIT: Transient Causal Interaction.

This sub-package contains helper functions organized into logical modules:

Core Utilities (core.py):
    - Event extraction and windowing
    - Statistics computation for VAR models
    - DeSnap analysis for event-based causality

Signal Processing (signal.py):
    - Peak detection and alignment
    - Location resampling and optimization
    - Signal windowing and extraction

Preprocessing (preprocess.py):
    - Artifact removal and trial filtering
    - Matrix regularization for numerical stability
    - Data validation and cleaning

Residual Computation (residuals.py):
    - VAR model residual estimation
    - Residual covariance computation
    - Residual bias calculation

Helper Functions (helpers.py):
    - Covariance matrix computation
    - Coefficient estimation for VAR models
    - Multi-variable linear regression

Plotting (plotting.py):
    - Visualization utilities for causality results
    - Standard deviation plotting with shading
    - Matplotlib integration (optional)

All functions are designed to work with the TranCIT pipeline and provide
robust, well-documented utilities for neuroscience time series analysis.

Example
-------
>>> import numpy as np
>>> from trancit.utils import extract_event_snapshots, compute_event_statistics
>>>
>>> # Generate sample data
>>> signal = np.random.randn(2, 1000)  # (n_vars, time)
>>> locations = np.array([100, 200, 300])  # Event locations
>>>
>>> # Extract event snapshots
>>> snapshots = extract_event_snapshots(
...     signal, locations, model_order=4, lag_step=1,
...     start_offset=50, extract_length=100
... )
>>>
>>> # Compute statistics
>>> stats = compute_event_statistics(snapshots, model_order=4)
>>> print(f"Snapshots shape: {snapshots.shape}")
>>> print(f"Statistics keys: {list(stats.keys())}")
"""

# Core utilities - Event extraction and statistics
from .core import (
    compute_event_statistics,
    extract_event_snapshots,
    extract_event_windows,
    perform_desnap_analysis,
)

# Helper functions - Covariance and coefficient estimation
from .helpers import (
    compute_covariances,
    compute_multi_variable_linear_regression,
    estimate_coefficients,
)

# Preprocessing utilities - Data cleaning and validation
from .preprocess import regularize_if_singular, remove_artifact_trials

# Residual computation - VAR model residuals
from .residuals import estimate_residuals, get_residuals

# Signal processing - Peak detection and alignment
from .signal import (
    find_best_shrinked_locations,
    find_peak_locations,
    shrink_locations_resample_uniform,
)

# Optional plotting utilities
try:
    from .plotting import fill_std_known

    _plotting_available = True
except ImportError:
    _plotting_available = False

    # Create a placeholder function for when matplotlib is not available
    def fill_std_known(*args, **kwargs):
        """Placeholder function when matplotlib is not available."""
        raise ImportError(
            "matplotlib is required for plotting functions. "
            "Install with: pip install matplotlib"
        )


# Public API definition
__all__ = [
    # Core utilities
    "extract_event_snapshots",
    "compute_event_statistics",
    "perform_desnap_analysis",
    "extract_event_windows",
    # Preprocessing
    "remove_artifact_trials",
    "regularize_if_singular",
    # Residual computation
    "get_residuals",
    "estimate_residuals",
    # Signal processing
    "find_peak_locations",
    "shrink_locations_resample_uniform",
    "find_best_shrinked_locations",
    # Helper functions
    "compute_covariances",
    "estimate_coefficients",
    "compute_multi_variable_linear_regression",
    # Plotting utilities
    "fill_std_known",
]

# Note: __all__ is defined above and includes all public functions
