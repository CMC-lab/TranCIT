import numpy as np
from dcs.simulation import generate_signals
# Assuming dataclasses are defined in dcs.pipeline or dcs.config
from dcs.pipeline import (
    snapshot_detect_analysis_pipeline,
    PipelineConfig, PipelineOptions, DetectionParams, BicParams, CausalParams, OutputParams, MonteCParams
)

# --- 1. Generate Data ---
# (Same as before)
T = 1200; Ntrial = 20; h = 0.1; gamma1=0.5; gamma2=0.5; Omega1=1.0; Omega2=1.2
data, _, _ = generate_signals(T, Ntrial, h, gamma1, gamma2, Omega1, Omega2)
original_signal_for_pipeline = np.mean(data, axis=2)
detection_signal_for_pipeline = original_signal_for_pipeline

# --- 2. Create Configuration Object ---
# Instantiate the dataclasses to build the config

config = PipelineConfig(
    options=PipelineOptions(
        detection=True,
        bic=False,
        causal_analysis=True,
        bootstrap=False, # Set to True to enable bootstrapping
        save_flag=False
    ),
    detection=DetectionParams(
        thres_ratio=2.0,
        align_type='peak',
        l_extract=150,
        l_start=75,
        remove_artif=True
        # locs is not needed since detection=True
    ),
    bic=BicParams(
        morder=4 # Required even if bic=False
        # momax, tau, mode not needed if bic=False
    ),
    causal=CausalParams(
        ref_time=75,
        estim_mode='OLS'
    ),
    monte_carlo=MonteCParams( # Provide even if bootstrap=False, or handle None
         n_btsp=50 # Number of bootstrap samples if enabled
    ),
    output=OutputParams(
        file_keyword="notebook_example_run"
    )
)

# --- 3. Run Pipeline with Config Object ---
try:
    # Pass the config object directly
    snap_output, final_config, event_snapshots = snapshot_detect_analysis_pipeline(
        original_signal=original_signal_for_pipeline,
        detection_signal=detection_signal_for_pipeline,
        config=config
    )

    # --- 4. Use Results ---
    print("Pipeline finished.")
    if snap_output.get('CausalOutput'):
        print("DCS result shape:", snap_output['CausalOutput']['OLS']['DCS'].shape)

except Exception as e:
    print(f"An error occurred: {e}")
