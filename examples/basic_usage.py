# examples/basic_usage.py
"""
Basic Usage Example for the Dynamic Causal Strength (DCS) Package.

This script demonstrates:
1. Generating synthetic bivariate time series data.
2. Setting up parameters for the analysis pipeline.
3. Running the snapshot detection and causality analysis pipeline.
4. Accessing and printing some of the key results.
"""

import numpy as np
import logging

# Assuming 'dcs' package is installed or accessible in the Python path
try:
    from dcs.simulation import generate_signals
    from dcs.pipeline import snapshot_detect_analysis_pipeline
except ImportError as e:
    print(f"Error importing dcs package: {e}")
    print("Ensure the 'dcs' package is installed correctly.")
    exit()

# Configure logging format and level
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

def main():
    """Main function to run the basic usage example."""

    # --- 1. Generate Synthetic Data ---
    logging.info("Generating synthetic data...")
    # Simulation parameters (adjust as needed)
    T = 1200  # Number of time points
    Ntrial = 20 # Number of trials
    h = 0.1   # Time step
    gamma1, gamma2 = 0.5, 0.5 # Damping
    Omega1, Omega2 = 1.0, 1.2 # Frequencies

    # Generate signals (shape: [variables, time_points, trials])
    # Note: generate_signals uses T points but returns T-500 points
    try:
        data, _, _ = generate_signals(T, Ntrial, h, gamma1, gamma2, Omega1, Omega2)
        logging.info(f"Generated data shape: {data.shape}")
    except Exception as e:
        logging.error(f"Failed to generate signals: {e}")
        return

    # Prepare inputs for the pipeline
    # The pipeline expects:
    # - original_signal: (n_vars, n_time_points) -> often trial-averaged data
    # - detection_signal: (2, n_time_points) -> signal used for event detection/alignment
    # Here, we'll use the trial-averaged data for both for simplicity.
    if data.size == 0:
        logging.error("Generated data is empty.")
        return
    original_signal_for_pipeline = np.mean(data, axis=2)
    detection_signal_for_pipeline = original_signal_for_pipeline # Using same signal

    # --- 2. Define Pipeline Parameters ---
    logging.info("Setting up pipeline parameters...")
    # This dictionary configures the entire analysis process.
    # Refer to snapshot_detect_analysis_pipeline docstring for details.
    pipeline_params = {
        'Options': {
            'Detection': 1,         # 1 = Detect events, 0 = Use predefined locs
            'BIC': False,           # Use fixed model order instead of BIC
            'CausalAnalysis': True, # Perform causality analysis
            'Bootstrap': False,     # Don't run bootstrapping
            'PSD': False,           # Don't run PSD analysis (assuming removed)
            'save_flag': False      # Don't save results to file in this example
        },
        'Detection': {
            'ThresRatio': 2.0,      # Threshold for event detection (std deviations)
            'AlignType': 'peak',    # Align events based on signal peak
            'L_extract': 150,       # Length (time points) of extracted event window
            'L_start': 75,          # Start extraction 75 points before alignment point
            'ShrinkFlag': False,    # Not used for 'peak' alignment
            'remove_artif': True,   # Attempt to remove artifact trials
            # 'locs': np.array([...]) # Only needed if Options['Detection'] == 0
        },
        'BIC': {
            'morder': 4             # Fixed VAR model order (since BIC=False)
            # 'momax': 10, 'tau': 1, 'mode': 'biased' # Needed if BIC=True
        },
        'CausalParams': {           # Parameters for time_varying_causality
            'ref_time': 75,         # Reference time point for rDCS calculation
            'estim_mode': 'OLS',    # Estimation mode ('OLS' or 'RLS')
            'diag_flag': False,     # Use full covariance for DCS calculation
            'old_version': False    # Use updated rDCS calculation
        },
        'MonteC_Params': {},        # Not used if Bootstrap=False
        'PSD': {},                  # Not used if PSD=False
        'Output': {
            'FileKeyword': 'basic_usage_example' # Base name if saving were enabled
        }
    }

    # --- 3. Execute the Pipeline ---
    logging.info("Running the analysis pipeline...")
    try:
        # The pipeline returns main outputs, updated params, and the event snapshots
        snap_output, updated_params, event_snapshots = snapshot_detect_analysis_pipeline(
            original_signal=original_signal_for_pipeline,
            detection_signal=detection_signal_for_pipeline,
            params=pipeline_params
        )
        logging.info("Pipeline execution finished.")

    except (ValueError, TypeError, KeyError) as e:
        # Catch potential errors from input validation or processing
        logging.error(f"Pipeline failed: {e}")
        return
    except Exception as e:
        # Catch any other unexpected errors
        logging.error(f"An unexpected error occurred: {e}")
        return

    # --- 4. Inspect Results ---
    logging.info("Inspecting pipeline results...")

    # Check detected event locations
    if 'locs' in snap_output:
        logging.info(f"Detected {len(snap_output['locs'])} event locations.")
        # print(f"Locations: {snap_output['locs']}") # Uncomment to see locations

    # Check model order used
    if 'morder' in snap_output:
        logging.info(f"VAR model order used: {snap_output['morder']}")

    # Check causality results (if analysis was run)
    if snap_output.get('CausalOutput'):
        causal_results = snap_output['CausalOutput']['OLS'] # Assuming OLS mode
        logging.info("Causality analysis output available.")

        if 'DCS' in causal_results:
            dcs_data = causal_results['DCS']
            print(f"\nShape of Dynamic Causal Strength (DCS) array: {dcs_data.shape}")
            # Plotting dcs_data[:, 0] and dcs_data[:, 1] vs time would show results
            # (Requires matplotlib)
            # import matplotlib.pyplot as plt
            # plt.plot(dcs_data)
            # plt.title('Dynamic Causal Strength')
            # plt.xlabel('Time within window')
            # plt.ylabel('DCS')
            # plt.legend(['Y->X', 'X->Y'])
            # plt.show()
        else:
            logging.warning("DCS results not found in CausalOutput.")

        # You can similarly access 'TE' (Transfer Entropy) and 'rDCS' if needed
        # print(f"Shape of Transfer Entropy (TE) array: {causal_results.get('TE', np.array([])).shape}")

    else:
        logging.info("Causality analysis was not performed or did not produce output.")

    # You can also inspect 'Yt_stats' for model coefficients, covariances etc.
    # if 'Yt_stats' in snap_output:
    #    logging.info("Model statistics ('Yt_stats') are available.")


if __name__ == "__main__":
    main()
