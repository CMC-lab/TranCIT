import os

import numpy as np
import scipy.io as sio
from dcs import PipelineOrchestrator
from dcs.config import (BicParams, CausalParams, DeSnapParams, DetectionParams,
                        MonteCParams, OutputParams, PipelineConfig,
                        PipelineOptions)
from scipy.signal import firwin, lfilter


def _remove_boundary_locations(y, locs, L_start, L_event):
    """Remove boundary locations that would cause out-of-bounds extraction."""
    locs = locs[locs >= L_start]
    locs = locs[locs <= len(y) - L_event]
    return locs


config = PipelineConfig(
    options=PipelineOptions(
        detection=True,        # Perform detection in pipeline
        bic=True,              # Enable BIC model selection
        causal_analysis=True,  # Enable Causality analysis
        bootstrap=False,       # Enable bootstrapping
        debiased_stats=False,   # Disable DeSnap in this example
        save_flag=False,         # Set True to align
    ),
    detection=DetectionParams(
        l_start=200,           # Start offset for snapshot extraction
        l_extract=401,         # Length of extracted snapshots
        thres_ratio=5.0,       # Threshold ratio for event detection
        align_type='peak',     # Align to peaks
        locs=np.array([]) if True else None,  # Conditional on options.detection
        shrink_flag=False,     # Disable shrinking
        remove_artif=False     # Disable artifact removal
    ),
    bic=BicParams(
        morder=7,              # Default model order
        momax=10,              # Max order for BIC
        tau=1,                 # Lag step for snapshot extraction in BIC
        mode='biased',         # BIC mode
        estim_mode='OLS'       # Ordinary Least Squares estimation
    ),
    causal=CausalParams(
        old_version=False,     # Use new version of rDCS calculation
        diag_flag=False,        # Disable diagonal covariance approximation
        ref_time=range(1, 201),  # Reference time range (middle of pre-alignment window)
        estim_mode='OLS'       # Ordinary Least Squares estimation
    ),
    output=OutputParams(
        file_keyword="example_lfp_run",  # Base keyword for output files
        save_path=""  # Empty path for this example
    ),
    monte_carlo=MonteCParams(
        n_btsp=100  # Number of bootstrap samples
    ),
    desnap=DeSnapParams(
        detection_signal=None,
        original_signal=None,
        event_stats=None,
        morder=None,
        tau=None,
        l_start=None,
        l_extract=None,
        maxStdRatio=7.0,
        d0_max=None,
        d0=0.0,
        N_d=50
    )
)

print("--- Pipeline Configuration ---")
print(config)

event_rdcs_dir = "/Users/sali/Documents/Projects/cmcLab/repos/rtdac/localdata/causal_analysis/event_rdcs/"
sess_name = "vvp01_2006-4-9_18-43-47"
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
    ca1_lfp = ca1_data["CA1_lfp"]
    print(f"Loaded CA1 LFP shape: {ca1_lfp.shape}")
except FileNotFoundError:
    print(f"Error: CA1 file not found at {ca1_mat_path}")
    exit()
except KeyError:
    print(f"Error: Key 'CA1_lfp' not found in {ca1_mat_path}")
    exit()

ca1_channels = np.arange(0, 4)
ca3_channels = np.arange(0, 4)

filter_order = 49
B = firwin(
    numtaps=filter_order + 1,
    cutoff=[140, 230],  # Default passband
    pass_zero=False,
    window='hamming',
    fs=1252  # Default sampling rate
)

delay = filter_order // 2

num_total_channel_pairs = len(ca3_channels) * len(ca1_channels)
print(f"The total number of channel pairs: {num_total_channel_pairs}")

num_channel_pairs_processed = 0

orchestrator = PipelineOrchestrator(config)

for i in ca3_channels:
    for j in ca1_channels:
        print(f"Processing: {num_channel_pairs_processed}/{num_total_channel_pairs}")
        
        y = np.vstack((ca3_lfp[:, i], ca1_lfp[:, j]))
        
        yf = lfilter(B, 1, y.squeeze(), axis=-1)
        yf = yf[..., delay:]
        
        temp_idx = np.where(yf[1, :] >= np.mean(yf[1, :]) + config.detection.thres_ratio * np.std(yf[1, :]))[0]
        temp_idx = _remove_boundary_locations(yf[1, :], temp_idx, config.detection.l_start, config.detection.l_extract)
        
        yf_labels = np.zeros(yf[0, :].shape)
        
        for idx in temp_idx:
            start_idx = idx - config.detection.l_start
            end_idx = start_idx + config.detection.l_extract
            
            if start_idx >= 0 and end_idx <= len(yf_labels):
                yf_labels[start_idx:end_idx] = np.ones(config.detection.l_extract)
        
        for channel in ['ca3', 'ca1']:
            aligned_signal = yf[0 if channel == 'ca3' else 1, :] * yf_labels
            
            config.output.file_keyword = os.path.join(config.output.save_path,
                                                f"ca3_ca1_{sess_name}_chpair_{num_channel_pairs_processed}_{channel}_{config.detection.align_type}")
            
            try:
                detection_signal_input = np.vstack((aligned_signal, aligned_signal))
                
                result = orchestrator.run(
                    original_signal=yf,
                    detection_signal=detection_signal_input
                )
                print(f"Pipeline completed for {channel} channel in pair {num_channel_pairs_processed}")
                
            except Exception as e:
                print(f"Pipeline Error for {channel} channel in pair {num_channel_pairs_processed}: {e}")
        
        num_channel_pairs_processed += 1
        print(f"Completed {num_channel_pairs_processed}/{num_total_channel_pairs} channel pairs")

print(f"\n--- Analysis Complete ---")
