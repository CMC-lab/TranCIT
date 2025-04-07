.. _quickstart:

##########
Quickstart
##########

Here's a quick example to get you started with detecting causal strength using the main pipeline. This example generates synthetic data and runs the analysis with basic parameters.

.. code-block:: python
   :linenos:

   import numpy as np
   # Assuming your package 'dcs' is installed or in the Python path
   from dcs.simulation import generate_signals
   # Assuming config classes are defined in dcs.pipeline or dcs.config
   from dcs.pipeline import (
       snapshot_detect_analysis_pipeline,
       PipelineConfig, PipelineOptions, DetectionParams, BicParams, CausalParams, OutputParams, MonteCParams
   )
   import logging

   # Configure logging to see INFO messages
   logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

   # --- 1. Generate some simple synthetic data ---
   logging.info("Generating synthetic data...")
   T = 1000  # Reduced length for quicker example
   Ntrial = 10
   h = 0.1
   gamma1, gamma2 = 0.5, 0.5
   Omega1, Omega2 = 1, 1

   # Generate bivariate signals (variables, time, trials)
   data, _, _ = generate_signals(T, Ntrial, h, gamma1, gamma2, Omega1, Omega2)
   # Prepare inputs for the pipeline
   original_signal_for_pipeline = np.mean(data, axis=2)
   detection_signal_for_pipeline = original_signal_for_pipeline

   # --- 2. Set up minimal configuration for the pipeline ---
   logging.info("Setting up pipeline configuration...")
   config = PipelineConfig(
       options=PipelineOptions(
           detection=True,
           bic=False,
           causal_analysis=True,
           bootstrap=False,
           save_flag=False
       ),
       detection=DetectionParams(
           thres_ratio=1.5,
           align_type='peak',
           l_extract=100,
           l_start=50,
           shrink_flag=False,
           remove_artif=False
       ),
       bic=BicParams(
           morder=3 # Use fixed model order if bic=False
       ),
       causal=CausalParams(
           ref_time=50,
           estim_mode='OLS'
       ),
       # monte_carlo can be None or omitted if bootstrap is False and validation allows
       monte_carlo=None, # Or MonteCParams() if needed by __post_init__
       output=OutputParams(
           file_keyword='quickstart_example'
       )
   )

   # --- 3. Run the analysis pipeline ---
   logging.info("Running the analysis pipeline...")
   try:
       snap_output, final_config, event_snapshots = snapshot_detect_analysis_pipeline(
           original_signal=original_signal_for_pipeline,
           detection_signal=detection_signal_for_pipeline,
           config=config
       )

       # --- 4. Display some results ---
       logging.info("Pipeline completed successfully.")
       if snap_output.get('CausalOutput'):
           dcs_results = snap_output['CausalOutput']['OLS']['DCS']
           print("\nCalculated Dynamic Causal Strength (DCS) snippet (first 5 time points):")
           print(dcs_results[:5, :])
           print(f"\nDCS array shape: {dcs_results.shape}")
       else:
           print("\nCausality analysis was not run or produced no output.")

   except Exception as e:
       logging.error(f"An error occurred during the pipeline execution: {e}", exc_info=True)
