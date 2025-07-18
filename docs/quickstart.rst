
.. _quickstart:

##########
Quickstart
##########

This quick example walks you through running the main causal strength analysis pipeline using synthetic data.

We'll use a built-in simulator and minimal configuration. For real-world applications, see the full examples in the `examples/` folder.

.. code-block:: python
   :linenos:

   import numpy as np
   import logging
   from dcs import generate_signals, snapshot_detect_analysis_pipeline
   from dcs.config import PipelineConfig, PipelineOptions, DetectionParams, BicParams, CausalParams, OutputParams

   # Enable logging to show pipeline steps
   logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

   # --- 1. Generate synthetic bivariate time-series data ---
   logging.info("Generating synthetic data...")
   T, Ntrial, h = 1000, 10, 0.1
   gamma1, gamma2, Omega1, Omega2 = 0.5, 0.5, 1.0, 1.0
   data, _, _ = generate_signals(T, Ntrial, h, gamma1, gamma2, Omega1, Omega2)

   # Prepare pipeline inputs (average over trials)
   original_signal = np.mean(data, axis=2)
   detection_signal = original_signal  # Use the same input for simplicity

   # --- 2. Set up pipeline configuration ---
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
       bic=BicParams(morder=3),
       causal=CausalParams(
           ref_time=range(1, 101),
           estim_mode='OLS'
       ),
       output=OutputParams(file_keyword='quickstart_example')
   )

   # --- 3. Run the pipeline ---
   logging.info("Running the analysis pipeline...")
   try:
       snap_output, final_config, event_snapshots = snapshot_detect_analysis_pipeline(
           original_signal=original_signal,
           detection_signal=detection_signal,
           config=config
       )

       # --- 4. Access and display DCS results ---
       if snap_output.get('CausalOutput'):
           dcs_result = snap_output['CausalOutput']['OLS']['DCS']
           print("DCS shape:", dcs_result.shape)
           print("DCS values (first 5 time points):\n", dcs_result[:5, :])
       else:
           print("No causal output was generated.")
   except Exception as e:
       logging.error(f"Pipeline failed: {e}")

What Next?
==========

- Try changing `thres_ratio` or `ref_time` in the config to explore their effects.
- Look at `examples/basic_usage.py` and `examples/lfp_pipeline.py` for advanced usage.
