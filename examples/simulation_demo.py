import numpy as np
# Assuming your package 'dcs' is installed or in the Python path
from dcs.simulation import generate_signals
from dcs.pipeline import snapshot_detect_analysis_pipeline
import logging

# Configure logging to see INFO messages
logging.basicConfig(level=logging.INFO)

# --- 1. Generate some simple synthetic data ---
# (Using parameters similar to the README example)
T = 1000  # Reduced length for quicker example
Ntrial = 10
h = 0.1
gamma1, gamma2 = 0.5, 0.5
Omega1, Omega2 = 1, 1

# Generate bivariate signals (variables, time, trials)
# Note: generate_signals returns T-500 points
data, _, _ = generate_signals(T, Ntrial, h, gamma1, gamma2, Omega1, Omega2)
# We need original_signal (n_vars, n_time_points) for pipeline
# Using mean across trials as a placeholder for original_signal
# and one channel as detection_signal
original_signal_for_pipeline = np.mean(data, axis=2)
detection_signal_for_pipeline = original_signal_for_pipeline # Use itself for detection

# --- 2. Set up minimal parameters for the pipeline ---
# (Referencing necessary params from pipeline.py and previous discussions)
# NOTE: Excludes PSD/Bootstrap options for simplicity. Adjust as needed.
pipeline_params = {
    'Options': {
        'Detection': 1,         # Perform detection
        'BIC': False,           # Don't run BIC for this simple example
        'CausalAnalysis': True, # Calculate causality
        'Bootstrap': False,
        'PSD': False,           # Assuming PSD related code was removed
        'save_flag': False
    },
    'Detection': {
        'ThresRatio': 1.5,      # Example threshold ratio
        'AlignType': 'peak',    # Align to peaks
        'L_extract': 100,       # Window length to extract
        'L_start': 50,          # Offset from peak for start of window
        'ShrinkFlag': False,
        'remove_artif': False   # Don't remove artifacts in this simple run
    },
    'BIC': {
        'morder': 3             # Use fixed model order if BIC=False
        # Other BIC params not needed if BIC=False
    },
    'CausalParams': {
        'ref_time': 50,         # Example reference time for rDCS
        'estim_mode': 'OLS',
        'diag_flag': False,
        'old_version': False
    },
    'MonteC_Params': {},        # Not used if Bootstrap=False
    'PSD': {},                  # Not used if PSD=False
    'Output': {
        'FileKeyword': 'quickstart_example' # Needed even if save_flag=False
    }
}

# --- 3. Run the analysis pipeline ---
try:
    snap_output, _, _ = snapshot_detect_analysis_pipeline(
        original_signal=original_signal_for_pipeline,
        detection_signal=detection_signal_for_pipeline,
        params=pipeline_params
    )

    # --- 4. Display some results ---
    logging.info("Pipeline completed successfully.")
    # Print the calculated dynamic causal strength (DCS) if available
    if snap_output.get('CausalOutput'):
        dcs_results = snap_output['CausalOutput']['OLS']['DCS']
        print("\nCalculated Dynamic Causal Strength (DCS) snippet (first 5 time points):")
        print(dcs_results[:5, :])
        print(f"\nDCS array shape: {dcs_results.shape}") # (L_extract, 2)
    else:
        print("\nCausality analysis was not run or produced no output.")

except Exception as e:
    logging.error(f"An error occurred during the pipeline execution: {e}")
