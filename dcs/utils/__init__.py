"""
Utilities Sub-package for DCS

This sub-package contains helper functions for signal processing,
preprocessing, core calculations, simulation helpers, and plotting.

Selected functions, likely most relevant for direct use or interpreting
pipeline results, are imported here for easier access. More specialized
functions can be imported directly from their respective submodules
(e.g., from dcs.utils.signal import find_peak_loc).
"""

from .core import compute_event_statistics, extract_event_snapshots
from .preprocess import remove_artifact_trials
from .residuals import get_residuals

try:
    from .plotting import fill_std_known
    _plotting_available = True
except ImportError:
    _plotting_available = False
    pass

_public_api = [
    'extract_event_snapshots',
    'compute_event_statistics',
    'remove_artifact_trials',
    'get_residuals',
]

if _plotting_available:
    _public_api.append('fill_std_known')

__all__ = _public_api
