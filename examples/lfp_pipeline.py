import os
from configparser import ConfigParser
from typing import Optional

import numpy as np
import scipy.io as sio
from scipy.signal import filtfilt, firwin

try:
    from dcs import snapshot_detect_analysis_pipeline
    from dcs.config import (BicParams, CausalParams, DetectionParams,
                            MonteCParams, OutputParams, PipelineConfig,
                            PipelineOptions)
except ImportError as e:
    print(f"Error importing dcs package: {e}")
    print("Ensure the package is installed and configured correctly.")
    exit()

# --- Define Pipeline Configuration ---
# Using the dataclasses defined in dcs.config or dcs.pipeline
pipeline_config = PipelineConfig(
    options=PipelineOptions(
        detection=True,      # Let the pipeline handle detection
        bic=True,            # Enable BIC
        causal_analysis=True,# Enable Causality
        bootstrap=True,      # Enable Bootstrap
        save_flag=False      # Don't save from pipeline in this example
    ),
    detection=DetectionParams(
        thres_ratio=5,       # Threshold for detection signal
        align_type="peak",   # Align detected events to signal peak
        l_extract=401,       # Length of snapshot window to extract
        l_start=200,         # Start window 200 samples before alignment point
        shrink_flag=False,   # Not used for peak alignment
        locs=None,           # Let pipeline find locs since detection=True
        remove_artif=False   # Disable artifact rejection for now
    ),
    bic=BicParams(
        morder=7,            # Default/fallback model order
        momax=10,            # Max order for BIC check
        tau=1,               # Lag step for BIC snapshot extraction
        mode="biased"        # BIC mode
    ),
    causal=CausalParams(
        old_version=False,
        diag_flag=False,
        ref_time=range(1, 101), # Example: Middle of pre-alignment window (if L_start=200)
        estim_mode="OLS"
    ),
    output=OutputParams(
        file_keyword="example_lfp_run" # Base keyword if saving were enabled
    ),
    monte_carlo=MonteCParams(
        n_btsp=100           # Number of bootstrap samples if bootstrap=True
    )
)

# --- Load Data ---
event_rdcs_dir = ""
sess_name = "example_session"
ca3_mat_path = os.path.join(event_rdcs_dir, f"{sess_name}_CA3.mat")
ca1_mat_path = os.path.join(event_rdcs_dir, f"{sess_name}_CA1.mat")

print(f"Loading CA3 data from: {ca3_mat_path}")
try:
    ca3_data = sio.loadmat(ca3_mat_path)
    ca3_lfp = ca3_data["CA3_lfp"]
    print(f"Loaded CA3 LFP shape: {ca3_lfp.shape}")
except FileNotFoundError:
    print(f"Error: CA3 file not found at {ca3_mat_path}")
    exit()
except KeyError:
    print(f"Error: Key 'CA3_lfp' not found in {ca3_mat_path}")
    exit()

print(f"Loading CA1 data from: {ca1_mat_path}")
try:
    ca1_data = sio.loadmat(ca1_mat_path)
    ca1_lfp = ca1_data["CA1_lfp"] # Assuming key is 'CA1_lfp'
    print(f"Loaded CA1 LFP shape: {ca1_lfp.shape}")
except FileNotFoundError:
    print(f"Error: CA1 file not found at {ca1_mat_path}")
    exit()
except KeyError:
    print(f"Error: Key 'CA1_lfp' not found in {ca1_mat_path}")
    exit()

# --- Channel Setup ---
ca1_channels = np.arange(0, 32) # Example: Channels 0-31
ca3_channels = np.arange(0, 8)  # Example: Channels 0-7
print(f"Processing CA1 channels: {ca1_channels}")
print(f"Processing CA3 channels: {ca3_channels}")

# --- Filtering Setup ---
Fs = 1252 # Example: Sampling frequency in Hz
passband = [140, 230] # Example: Passband in Hz
filter_order = 50 # Example: Filter order (even number needed for firwin type 1)

# Design the FIR filter coefficients
try:
    filter_coeffs = firwin(filter_order + 1, np.array(passband) / (0.5 * Fs), pass_zero=False)
    print(f"Designed FIR filter (Order: {filter_order}, Passband: {passband} Hz, Fs: {Fs} Hz)")
except NameError:
    print("Error: 'Fs' or 'passband' not defined. Cannot design filter.")
    exit()
except Exception as e:
    print(f"Error designing FIR filter: {e}")
    exit()

# --- Process Channel Pairs ---
results_all_pairs = {} # Dictionary to store results per pair
n_chpair_total = len(ca3_channels) * len(ca1_channels)
n_chpair_count = 0

print(f"\nStarting analysis for {n_chpair_total} channel pairs...")

for i in ca3_channels:
    for j in ca1_channels:
        n_chpair_count += 1
        pair_key = f"CA3_{i}-CA1_{j}"
        print(f"\n--- Processing Pair {n_chpair_count}/{n_chpair_total}: {pair_key} ---")

        # 1. Prepare bivariate signal for this pair
        if i >= ca3_lfp.shape[1] or j >= ca1_lfp.shape[1]:
            print(f"Warning: Channel index out of bounds for pair ({i}, {j}). Skipping.")
            continue
        y = np.vstack((ca3_lfp[:, i], ca1_lfp[:, j])) # Shape (2, L_signal)

        # 2. Filter the data (zero-phase)
        # Using filtfilt avoids phase distortion and manual delay compensation
        try:
            yf = filtfilt(filter_coeffs, 1.0, y, axis=1)
        except Exception as e:
            print(f"Error filtering data for pair {pair_key}: {e}. Skipping.")
            continue

        # 3. Run the pipeline
        # Pass the *filtered* data as BOTH original_signal and detection_signal
        # The pipeline will internally handle detection based on detection_signal[1,:]
        # (or adjust which channel to use for detection if needed via config or modification)
        print("Running snapshot_detect_analysis_pipeline...")
        try:
            # Assume detection should happen on the second channel (CA1)
            # If detection needed on CA3, swap order in detection_signal or adjust pipeline logic
            detection_signal_input = np.vstack((yf[0,:], yf[1,:])) # Example: Detect on CA1 (index 1)

            snapshot_output, _, _ = snapshot_detect_analysis_pipeline(
                original_signal=yf, # Use filtered data for analysis
                detection_signal=detection_signal_input, # Use filtered data for detection
                config=pipeline_config # Pass the configured object
            )
            print(f"Pipeline finished for {pair_key}.")
            # Store results for this pair
            results_all_pairs[pair_key] = snapshot_output

        except (ValueError, TypeError, KeyError) as e:
            print(f"Pipeline Error for pair {pair_key}: {e}. Skipping.")
        except Exception as e:
            print(f"Unexpected Pipeline Error for pair {pair_key}: {e}. Skipping.")

print(f"\n--- Analysis Complete ---")
print(f"Processed {len(results_all_pairs)} out of {n_chpair_total} channel pairs.")
