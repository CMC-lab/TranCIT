__license__ = "MIT"
__version__ = "0.1.0"
__author__ = "Salar Nouri"
__email__ = "salr.nouri@gmail.com"

from .causality import (compute_causal_strength_nonzero_mean,
                        time_varying_causality)
from .logger_config import setup_logging
from .models import (compute_BIC_for_model, compute_multi_trial_BIC,
                     estimate_var_coefficients, select_model_order)
from .pipeline import snapshot_detect_analysis_pipeline
from .simulation import (generate_signals, simulate_ar_event,
                         simulate_ar_event_bootstrap)

setup_logging(log_file="dcs_log.txt")

__all__ = ["causality", "config", "models", "pipeline", "simulation", "utils"]
