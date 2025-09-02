.. _tutorials:

#########
Tutorials
#########

This section provides comprehensive tutorials for using **TranCIT: Transient Causal Interaction Toolbox** in different scenarios. Each tutorial builds on the previous ones and includes complete working examples.

.. contents:: Tutorial Contents
   :local:
   :depth: 2

****************************
Tutorial 1: Basic DCS Analysis
****************************

Learn the fundamentals of causality analysis with Dynamic Causal Strength (DCS).

Understanding the Data Format
=============================

DCS works with multivariate time series data in a specific format:

.. code-block:: python

   import numpy as np
   from trancit import generate_signals
   
   # Generate sample data
   data, _, _ = generate_signals(
       T=500,        # Time points
       Ntrial=10,    # Number of trials/repetitions
       h=0.1,        # Time step
       gamma1=0.5,   # Damping parameter 1
       gamma2=0.5,   # Damping parameter 2  
       Omega1=1.0,   # Natural frequency 1
       Omega2=1.2    # Natural frequency 2
   )
   
   print(f"Data shape: {data.shape}")
   # Output: (2, 500, 10) = (variables, time_points, trials)
   
   # Visualize the data structure
   print(f"Variable 1 shape: {data[0].shape}")  # (500, 10)
   print(f"Variable 2 shape: {data[1].shape}")  # (500, 10)
   
   # Each trial is a realization of the same underlying system
   print(f"Trial 0 of variable 1: {data[0, :5, 0]}")  # First 5 time points
   print(f"Trial 1 of variable 1: {data[0, :5, 1]}")  # Same time points, different trial

**Key Points:**

- **Shape**: ``(n_variables, n_timepoints, n_trials)``
- **Variables**: DCS analyzes bivariate relationships (n_variables = 2)
- **Trials**: Multiple realizations improve statistical robustness
- **Time**: Each time point represents one measurement

Step-by-Step DCS Analysis
==========================

.. code-block:: python

   from trancit import DCSCalculator
   import matplotlib.pyplot as plt
   
   # Step 1: Create calculator with appropriate model order
   calculator = DCSCalculator(
       model_order=4,          # Number of time lags to consider
       time_mode="inhomo",     # Time-inhomogeneous analysis (recommended)
       use_diagonal_covariance=False  # Use full covariance (more accurate)
   )
   
   # Step 2: Perform the analysis
   result = calculator.analyze(data)
   
   # Step 3: Examine the results
   print(f"Analysis completed!")
   print(f"Causal strength shape: {result.causal_strength.shape}")
   print(f"Transfer entropy shape: {result.transfer_entropy.shape}")
   print(f"Granger causality shape: {result.granger_causality.shape}")
   
   # Step 4: Interpret the results
   # Column 0: Y → X (variable 2 influences variable 1)  
   # Column 1: X → Y (variable 1 influences variable 2)
   
   mean_dcs_x_to_y = result.causal_strength[:, 1].mean()
   mean_dcs_y_to_x = result.causal_strength[:, 0].mean()
   
   print(f"Mean causal strength X→Y: {mean_dcs_x_to_y:.4f}")
   print(f"Mean causal strength Y→X: {mean_dcs_y_to_x:.4f}")
   
   if mean_dcs_x_to_y > mean_dcs_y_to_x:
       print("Stronger causality: X → Y")
   else:
       print("Stronger causality: Y → X")

Visualizing Results
===================

.. code-block:: python

   # Create comprehensive visualization
   fig, axes = plt.subplots(2, 2, figsize=(15, 10))
   
   # Plot 1: Original signals (first trial)
   axes[0, 0].plot(data[0, :100, 0], label='Variable X', alpha=0.8)
   axes[0, 0].plot(data[1, :100, 0], label='Variable Y', alpha=0.8)
   axes[0, 0].set_xlabel('Time')
   axes[0, 0].set_ylabel('Amplitude')
   axes[0, 0].set_title('Original Time Series (First 100 points)')
   axes[0, 0].legend()
   axes[0, 0].grid(True, alpha=0.3)
   
   # Plot 2: Dynamic Causal Strength over time
   axes[0, 1].plot(result.causal_strength[:, 1], label='X → Y', linewidth=2)
   axes[0, 1].plot(result.causal_strength[:, 0], label='Y → X', linewidth=2)
   axes[0, 1].set_xlabel('Time')
   axes[0, 1].set_ylabel('Causal Strength')
   axes[0, 1].set_title('Dynamic Causal Strength')
   axes[0, 1].legend()
   axes[0, 1].grid(True, alpha=0.3)
   
   # Plot 3: Transfer Entropy
   axes[1, 0].plot(result.transfer_entropy[:, 1], label='X → Y', linewidth=2)
   axes[1, 0].plot(result.transfer_entropy[:, 0], label='Y → X', linewidth=2)
   axes[1, 0].set_xlabel('Time')
   axes[1, 0].set_ylabel('Transfer Entropy')
   axes[1, 0].set_title('Transfer Entropy')
   axes[1, 0].legend()
   axes[1, 0].grid(True, alpha=0.3)
   
   # Plot 4: Comparison of measures
   time_points = range(len(result.causal_strength))
   width = 0.35
   
   mean_measures = {
       'DCS': [result.causal_strength[:, 0].mean(), result.causal_strength[:, 1].mean()],
       'TE': [result.transfer_entropy[:, 0].mean(), result.transfer_entropy[:, 1].mean()],
       'GC': [result.granger_causality[:, 0].mean(), result.granger_causality[:, 1].mean()]
   }
   
   x = np.arange(2)  # Y→X, X→Y
   axes[1, 1].bar(x - width, [mean_measures['DCS'][0], mean_measures['DCS'][1]], 
                  width, label='DCS', alpha=0.8)
   axes[1, 1].bar(x, [mean_measures['TE'][0], mean_measures['TE'][1]], 
                  width, label='TE', alpha=0.8)
   axes[1, 1].bar(x + width, [mean_measures['GC'][0], mean_measures['GC'][1]], 
                  width, label='GC', alpha=0.8)
   
   axes[1, 1].set_xlabel('Direction')
   axes[1, 1].set_ylabel('Mean Value')
   axes[1, 1].set_title('Comparison of Causality Measures')
   axes[1, 1].set_xticks(x)
   axes[1, 1].set_xticklabels(['Y → X', 'X → Y'])
   axes[1, 1].legend()
   axes[1, 1].grid(True, alpha=0.3)
   
   plt.tight_layout()
   plt.show()

**********************************************
Tutorial 2: Event-Based Analysis with Pipeline
**********************************************

Learn how to detect and analyze specific events in time series data.

Understanding Event Detection
=============================

Event-based analysis focuses on detecting specific time windows of interest and analyzing causality within those windows.

.. code-block:: python

   import numpy as np
   from trancit import generate_signals, PipelineOrchestrator
   from trancit.config import (
       PipelineConfig, PipelineOptions, DetectionParams, 
       CausalParams, BicParams, OutputParams
   )
   
   # Generate data with clearer event structure
   np.random.seed(42)
   data, _, _ = generate_signals(
       T=800, Ntrial=15, h=0.1,
       gamma1=0.3, gamma2=0.7,  # Asymmetric damping
       Omega1=0.8, Omega2=1.4   # Different frequencies
   )
   
   # Prepare signals for event detection
   original_signal = np.mean(data, axis=2)  # Average over trials
   
   # Create detection signal with enhanced events
   detection_signal = original_signal.copy()
   
   # Add some artificial "events" for demonstration
   event_times = [200, 400, 600]
   for t in event_times:
       # Enhance signal at event times
       detection_signal[:, t-10:t+10] *= 2.0
       detection_signal[:, t-5:t+5] += np.random.randn(2, 10) * 0.5
   
   print(f"Original signal shape: {original_signal.shape}")
   print(f"Detection signal shape: {detection_signal.shape}")

Configuring the Pipeline
=========================

The pipeline configuration controls every aspect of the analysis:

.. code-block:: python

   # Create comprehensive pipeline configuration
   config = PipelineConfig(
       # Main analysis options
       options=PipelineOptions(
           detection=True,          # Enable event detection
           bic=True,               # Enable BIC model selection
           causal_analysis=True,    # Enable causality analysis
           bootstrap=False,         # Skip bootstrap for speed
           save_flag=False,         # Don't save intermediate files
           debiased_stats=False     # Skip debiased analysis for speed
       ),
       
       # Event detection parameters
       detection=DetectionParams(
           thres_ratio=1.8,         # Threshold for event detection (lower = more events)
           align_type="peak",       # Align events to their peaks
           l_extract=100,           # Length of extracted event windows
           l_start=50,              # Starting point within extracted windows
           shrink_flag=False,       # Don't apply shrinkage
           remove_artif=True,       # Remove artifact-contaminated trials
           locs=None                # Automatically detect event locations
       ),
       
       # BIC model selection parameters
       bic=BicParams(
           morder=4,                # Default model order
           momax=8,                 # Maximum model order to test
           mode="OLS",              # Ordinary Least Squares
           tau=1                    # Smoothing parameter
       ),
       
       # Causality analysis parameters
       causal=CausalParams(
           ref_time=50,             # Reference time for rDCS (should match l_start)
           estim_mode="OLS",        # Estimation method
           diag_flag=False,         # Use full covariance matrix
           old_version=False        # Use new rDCS calculation method
       ),
       
       # Output parameters
       output=OutputParams(
           file_keyword="tutorial_events"
       )
   )
   
   print("Configuration created successfully")

Running the Event-Based Analysis
=================================

.. code-block:: python

   # Create and run the pipeline orchestrator
   orchestrator = PipelineOrchestrator(config)
   
   try:
       print("Starting pipeline analysis...")
       
       # Run the complete pipeline
       result = orchestrator.run(original_signal, detection_signal)
       
       print("Pipeline completed successfully!")
       
       # Examine the pipeline results
       print(f"Event snapshots shape: {result.event_snapshots.shape}")
       
       if result.results.get('locs') is not None:
           detected_events = result.results['locs']
           print(f"Number of events detected: {len(detected_events)}")
           print(f"Event locations: {detected_events}")
       
       # Access causality results if available
       if result.results.get("CausalOutput"):
           causal_output = result.results["CausalOutput"]["OLS"]
           
           if "DCS" in causal_output:
               dcs_results = causal_output["DCS"]
               print(f"DCS results shape: {dcs_results.shape}")
               print(f"Mean DCS X→Y: {dcs_results[:, 1].mean():.4f}")
               print(f"Mean DCS Y→X: {dcs_results[:, 0].mean():.4f}")
               
               # Analyze individual events
               print("\nPer-event analysis:")
               for i, (dcs_xy, dcs_yx) in enumerate(dcs_results):
                   print(f"Event {i+1}: X→Y={dcs_xy:.4f}, Y→X={dcs_yx:.4f}")
           
           if "rDCS" in causal_output:
               rdcs_results = causal_output["rDCS"]
               print(f"Relative DCS shape: {rdcs_results.shape}")
               print(f"Mean rDCS X→Y: {rdcs_results[:, 1].mean():.4f}")
               print(f"Mean rDCS Y→X: {rdcs_results[:, 0].mean():.4f}")
       
       else:
           print("No causal output generated")
           print("This might happen if no events were detected or analysis failed")
   
   except Exception as e:
       print(f"Pipeline analysis failed: {e}")
       print("Common issues:")
       print("1. No events detected - try lowering thres_ratio")
       print("2. Insufficient data - try shorter l_extract or more data")
       print("3. Numerical issues - try different model parameters")

Visualizing Event-Based Results
================================

.. code-block:: python

   # Create visualization of event-based analysis
   if 'result' in locals() and result.results.get("CausalOutput"):
       fig, axes = plt.subplots(3, 1, figsize=(15, 12))
       
       # Plot 1: Original signals with detected events
       time_axis = np.arange(original_signal.shape[1])
       axes[0].plot(time_axis, original_signal[0], label='Variable X', alpha=0.7)
       axes[0].plot(time_axis, original_signal[1], label='Variable Y', alpha=0.7)
       
       # Mark detected events
       if result.results.get('locs') is not None:
           for loc in result.results['locs']:
               axes[0].axvline(x=loc, color='red', linestyle='--', alpha=0.7)
               axes[0].text(loc, axes[0].get_ylim()[1]*0.9, 'Event', 
                           rotation=90, fontsize=8)
       
       axes[0].set_xlabel('Time')
       axes[0].set_ylabel('Amplitude')
       axes[0].set_title('Original Signals with Detected Events')
       axes[0].legend()
       axes[0].grid(True, alpha=0.3)
       
       # Plot 2: Event-wise causality measures
       if "DCS" in result.results["CausalOutput"]["OLS"]:
           dcs_data = result.results["CausalOutput"]["OLS"]["DCS"]
           event_nums = range(1, len(dcs_data) + 1)
           
           width = 0.35
           x = np.arange(len(event_nums))
           
           axes[1].bar(x - width/2, dcs_data[:, 0], width, 
                      label='Y → X', alpha=0.8, color='blue')
           axes[1].bar(x + width/2, dcs_data[:, 1], width,
                      label='X → Y', alpha=0.8, color='red')
           
           axes[1].set_xlabel('Event Number')
           axes[1].set_ylabel('Dynamic Causal Strength')
           axes[1].set_title('Per-Event Causality Analysis')
           axes[1].set_xticks(x)
           axes[1].set_xticklabels(event_nums)
           axes[1].legend()
           axes[1].grid(True, alpha=0.3)
       
       # Plot 3: Comparison of measures across events
       if ("DCS" in result.results["CausalOutput"]["OLS"] and 
           "TE" in result.results["CausalOutput"]["OLS"]):
           
           dcs_data = result.results["CausalOutput"]["OLS"]["DCS"]
           te_data = result.results["CausalOutput"]["OLS"]["TE"]
           
           # Plot X→Y direction
           axes[2].plot(dcs_data[:, 1], 'o-', label='DCS (X→Y)', linewidth=2)
           axes[2].plot(te_data[:, 1], 's-', label='TE (X→Y)', linewidth=2)
           
           axes[2].set_xlabel('Event Number')
           axes[2].set_ylabel('Causality Measure')
           axes[2].set_title('Causality Measures Across Events (X→Y Direction)')
           axes[2].legend()
           axes[2].grid(True, alpha=0.3)
       
       plt.tight_layout()
       plt.show()

*************************************************
Tutorial 3: Advanced Configuration and Optimization
*************************************************

Learn how to optimize DCS analysis for different types of data and research questions.

Model Order Selection
=====================

Choosing the right model order is crucial for accurate causality analysis:

.. code-block:: python

   from trancit import DCSCalculator
   from trancit.models import BICSelector
   import matplotlib.pyplot as plt
   
   # Generate test data
   np.random.seed(123)
   data, _, _ = generate_signals(T=600, Ntrial=20, h=0.1, 
                                gamma1=0.4, gamma2=0.6, 
                                Omega1=1.1, Omega2=0.9)
   
   # Method 1: Manual comparison of different model orders
   model_orders = [1, 2, 3, 4, 5, 6, 7, 8]
   causality_results = {}
   
   print("Testing different model orders...")
   for order in model_orders:
       try:
           calculator = DCSCalculator(model_order=order, time_mode="inhomo")
           result = calculator.analyze(data)
           
           causality_results[order] = {
               'dcs_mean': result.causal_strength.mean(axis=0),
               'te_mean': result.transfer_entropy.mean(axis=0),
               'gc_mean': result.granger_causality.mean(axis=0)
           }
           print(f"Order {order}: DCS X→Y = {result.causal_strength[:, 1].mean():.4f}")
           
       except Exception as e:
           print(f"Order {order} failed: {e}")
           causality_results[order] = None
   
   # Method 2: Automatic BIC-based selection
   print("\nUsing BIC for automatic model selection...")
   try:
       bic_selector = BICSelector(max_order=10, mode="OLS")
       
       # Prepare data for BIC analysis (requires specific format)
       bic_data = data.copy()  # Shape: (2, T, N)
       
       # BIC analysis parameters
       bic_params = {
           "Params": {
               "BIC": {
                   "momax": 8,
                   "mode": "OLS"
               }
           },
           "EstimMode": "OLS"
       }
       
       # Run BIC analysis
       bic_results = bic_selector.compute_multi_trial_BIC(bic_data, bic_params)
       
       if bic_results and 'morder' in bic_results:
           optimal_order = bic_results['morder']
           print(f"BIC-selected optimal model order: {optimal_order}")
           
           # Use optimal order for final analysis
           calculator = DCSCalculator(model_order=optimal_order, time_mode="inhomo")
           final_result = calculator.analyze(data)
           
           print(f"Final analysis with order {optimal_order}:")
           print(f"DCS X→Y: {final_result.causal_strength[:, 1].mean():.4f}")
           print(f"DCS Y→X: {final_result.causal_strength[:, 0].mean():.4f}")
       
   except Exception as e:
       print(f"BIC selection failed: {e}")
       print("Using default model order 4")
       optimal_order = 4

Handling Different Data Types
=============================

.. code-block:: python

   from trancit.utils.preprocess import normalize_data, check_data_quality
   
   # Simulate different types of real-world data issues
   np.random.seed(456)
   
   # 1. Noisy data
   print("1. Handling noisy data:")
   noisy_data, _, _ = generate_signals(T=400, Ntrial=15, h=0.1,
                                      gamma1=0.5, gamma2=0.5,
                                      Omega1=1.0, Omega2=1.2)
   # Add significant noise
   noisy_data += np.random.randn(*noisy_data.shape) * 0.5
   
   # Check data quality
   quality_issues = check_data_quality(noisy_data)
   print(f"Data quality issues: {quality_issues}")
   
   # Normalize to improve analysis
   clean_data = normalize_data(noisy_data, method="zscore", axis=1)
   
   # Compare results
   calc = DCSCalculator(model_order=4)
   
   try:
       result_noisy = calc.analyze(noisy_data)
       result_clean = calc.analyze(clean_data)
       
       print(f"Noisy data DCS X→Y: {result_noisy.causal_strength[:, 1].mean():.4f}")
       print(f"Clean data DCS X→Y: {result_clean.causal_strength[:, 1].mean():.4f}")
   except Exception as e:
       print(f"Analysis failed: {e}")
   
   # 2. Short time series
   print("\n2. Handling short time series:")
   short_data, _, _ = generate_signals(T=100, Ntrial=25, h=0.1,  # More trials, less time
                                      gamma1=0.5, gamma2=0.5,
                                      Omega1=1.0, Omega2=1.2)
   
   # Use smaller model order for short series
   calc_short = DCSCalculator(model_order=2, time_mode="homo")  # Homogeneous for short data
   
   try:
       result_short = calc_short.analyze(short_data)
       print(f"Short series DCS X→Y: {result_short.causal_strength[:, 1].mean():.4f}")
   except Exception as e:
       print(f"Short series analysis failed: {e}")
   
   # 3. Highly correlated data
   print("\n3. Handling highly correlated data:")
   T, N = 300, 20
   
   # Generate highly correlated signals
   base_signal = np.random.randn(T, N)
   corr_data = np.zeros((2, T, N))
   corr_data[0] = base_signal
   corr_data[1] = 0.95 * base_signal + 0.05 * np.random.randn(T, N)  # Very high correlation
   
   # Add regularization for numerical stability
   try:
       calc_reg = DCSCalculator(model_order=3, use_diagonal_covariance=True)
       result_corr = calc_reg.analyze(corr_data)
       print(f"Highly correlated data DCS X→Y: {result_corr.causal_strength[:, 1].mean():.4f}")
   except Exception as e:
       print(f"Correlated data analysis failed: {e}")

Performance Optimization
========================

.. code-block:: python

   import time
   from trancit import PipelineOrchestrator
   from trancit.config import PipelineConfig, PipelineOptions, DetectionParams, CausalParams
   
   # Generate larger dataset for performance testing
   large_data, _, _ = generate_signals(T=2000, Ntrial=30, h=0.1,
                                      gamma1=0.5, gamma2=0.5,
                                      Omega1=1.0, Omega2=1.2)
   
   original_signal = np.mean(large_data, axis=2)
   detection_signal = original_signal * 1.2
   
   print("Performance optimization comparison:")
   
   # Configuration 1: Full analysis (slower but comprehensive)
   full_config = PipelineConfig(
       options=PipelineOptions(
           detection=True,
           bic=True,              # BIC is computationally expensive
           causal_analysis=True,
           bootstrap=True,        # Bootstrap is time-consuming
           debiased_stats=True    # Additional computational overhead
       ),
       detection=DetectionParams(thres_ratio=2.0, l_extract=200, l_start=100),
       causal=CausalParams(ref_time=100, estim_mode="OLS"),
       # ... other params
   )
   
   # Configuration 2: Fast analysis (faster but less comprehensive)
   fast_config = PipelineConfig(
       options=PipelineOptions(
           detection=True,
           bic=False,             # Skip BIC for speed
           causal_analysis=True,
           bootstrap=False,       # Skip bootstrap for speed
           debiased_stats=False   # Skip debiased analysis for speed
       ),
       detection=DetectionParams(thres_ratio=2.0, l_extract=100, l_start=50),  # Shorter windows
       causal=CausalParams(ref_time=50, estim_mode="OLS"),
       # ... other params
   )
   
   # Time both approaches
   configurations = [
       ("Fast Configuration", fast_config),
       # ("Full Configuration", full_config)  # Uncomment for comparison
   ]
   
   for name, config in configurations:
       try:
           start_time = time.time()
           orchestrator = PipelineOrchestrator(config)
           result = orchestrator.run(original_signal, detection_signal)
           end_time = time.time()
           
           print(f"{name}: {end_time - start_time:.2f} seconds")
           
           if result.results.get("CausalOutput"):
               causal_data = result.results["CausalOutput"]["OLS"]
               if "DCS" in causal_data:
                   n_events = len(causal_data["DCS"])
                   print(f"  - Events detected: {n_events}")
                   print(f"  - Mean DCS X→Y: {causal_data['DCS'][:, 1].mean():.4f}")
           
       except Exception as e:
           print(f"{name} failed: {e}")

Statistical Validation
======================

.. code-block:: python

   from scipy import stats
   import numpy as np
   
   # Generate data with known causal structure for validation
   np.random.seed(789)
   
   # Create signals where X clearly influences Y
   T, N = 500, 25
   
   signal_x = np.random.randn(T, N)
   signal_y = np.zeros((T, N))
   
   # Y depends on past values of X (true causality X→Y)
   for t in range(2, T):
       signal_y[t] = (0.7 * signal_y[t-1] - 
                     0.2 * signal_y[t-2] + 
                     0.4 * signal_x[t-1] +     # Clear X→Y influence
                     0.1 * signal_x[t-2] +
                     np.random.randn(N) * 0.3)
   
   validation_data = np.array([signal_x.T, signal_y.T])
   
   # Multiple analyses for statistical validation
   n_bootstrap = 20
   dcs_xy_values = []
   dcs_yx_values = []
   
   print("Statistical validation with bootstrap sampling:")
   
   for i in range(n_bootstrap):
       # Random resampling of trials
       trial_indices = np.random.choice(N, size=N, replace=True)
       bootstrap_data = validation_data[:, :, trial_indices]
       
       calc = DCSCalculator(model_order=3, time_mode="inhomo")
       result = calc.analyze(bootstrap_data)
       
       dcs_xy_values.append(result.causal_strength[:, 1].mean())
       dcs_yx_values.append(result.causal_strength[:, 0].mean())
   
   # Statistical analysis
   dcs_xy_mean = np.mean(dcs_xy_values)
   dcs_yx_mean = np.mean(dcs_yx_values)
   dcs_xy_std = np.std(dcs_xy_values)
   dcs_yx_std = np.std(dcs_yx_values)
   
   print(f"DCS X→Y: {dcs_xy_mean:.4f} ± {dcs_xy_std:.4f}")
   print(f"DCS Y→X: {dcs_yx_mean:.4f} ± {dcs_yx_std:.4f}")
   
   # Statistical test for significant difference
   t_stat, p_value = stats.ttest_rel(dcs_xy_values, dcs_yx_values)
   
   print(f"Statistical test (paired t-test):")
   print(f"t-statistic: {t_stat:.4f}, p-value: {p_value:.4f}")
   
   if p_value < 0.05:
       if dcs_xy_mean > dcs_yx_mean:
           print("Significant causality detected: X → Y")
       else:
           print("Significant causality detected: Y → X")
   else:
       print("No significant causal asymmetry detected")

***************************************
Tutorial 4: Real-World Applications
***************************************

Apply DCS to realistic neuroscience and time series analysis scenarios.

Neural Data Analysis
====================

.. code-block:: python

   # Simulate Local Field Potential (LFP) data
   def simulate_lfp_data(duration=10.0, sampling_rate=1000, n_trials=30):
       """Simulate realistic LFP data with event-related responses."""
       
       n_samples = int(duration * sampling_rate)
       n_channels = 2
       
       # Base oscillatory activity
       t = np.linspace(0, duration, n_samples)
       
       data = np.zeros((n_channels, n_samples, n_trials))
       
       for trial in range(n_trials):
           # Base activity with multiple frequency components
           alpha_freq = 10 + np.random.randn() * 1  # 8-12 Hz alpha
           beta_freq = 20 + np.random.randn() * 3   # 15-25 Hz beta
           gamma_freq = 40 + np.random.randn() * 5  # 35-45 Hz gamma
           
           # Channel 1: Mix of frequencies
           data[0, :, trial] = (
               0.5 * np.sin(2 * np.pi * alpha_freq * t) +
               0.3 * np.sin(2 * np.pi * beta_freq * t) +
               0.2 * np.sin(2 * np.pi * gamma_freq * t) +
               np.random.randn(n_samples) * 0.1
           )
           
           # Channel 2: Influenced by Channel 1 with delay
           delay_samples = 5  # 5ms delay at 1kHz sampling
           data[1, delay_samples:, trial] = (
               0.6 * data[1, :-delay_samples, trial] +  # AR component
               0.4 * data[0, :-delay_samples, trial] +  # Channel 1 influence
               0.2 * np.sin(2 * np.pi * beta_freq * t[:-delay_samples]) +
               np.random.randn(n_samples - delay_samples) * 0.15
           )
           
           # Add event-related responses at random times
           n_events = np.random.poisson(3)  # Average 3 events per trial
           event_times = np.random.randint(1000, n_samples-1000, n_events)
           
           for event_time in event_times:
               # Event response in both channels
               event_duration = 200  # 200ms event
               event_window = slice(event_time, event_time + event_duration)
               
               # Enhanced coupling during events
               data[0, event_window, trial] *= 1.5
               data[1, event_window, trial] += 0.3 * data[0, event_window, trial]
       
       return data
   
   # Generate and analyze neural data
   print("Analyzing simulated neural data...")
   neural_data = simulate_lfp_data(duration=20.0, sampling_rate=500, n_trials=25)
   
   print(f"Neural data shape: {neural_data.shape}")
   print(f"Sampling rate: 500 Hz, Duration: 20s, Channels: 2, Trials: 25")
   
   # Preprocess neural data
   from trancit.utils.preprocess import normalize_data
   neural_data_norm = normalize_data(neural_data, method="zscore", axis=1)
   
   # DCS analysis with parameters suitable for neural data
   neural_calculator = DCSCalculator(
       model_order=6,             # Higher order for complex neural dynamics
       time_mode="inhomo",        # Non-stationary neural activity
       use_diagonal_covariance=False
   )
   
   try:
       neural_result = neural_calculator.analyze(neural_data_norm)
       
       print("Neural DCS Analysis Results:")
       print(f"Mean causality Ch1→Ch2: {neural_result.causal_strength[:, 1].mean():.4f}")
       print(f"Mean causality Ch2→Ch1: {neural_result.causal_strength[:, 0].mean():.4f}")
       print(f"Mean transfer entropy Ch1→Ch2: {neural_result.transfer_entropy[:, 1].mean():.4f}")
       
       # Identify periods of high causality
       high_causality_threshold = np.percentile(neural_result.causal_strength[:, 1], 90)
       high_causality_times = np.where(neural_result.causal_strength[:, 1] > high_causality_threshold)[0]
       
       print(f"High causality periods (top 10%): {len(high_causality_times)} time points")
       print(f"High causality times (first 10): {high_causality_times[:10]}")
       
   except Exception as e:
       print(f"Neural analysis failed: {e}")

Economic Time Series Analysis
=============================

.. code-block:: python

   # Simulate economic time series data
   def simulate_economic_data(n_days=1000, n_series=2):
       """Simulate economic time series (e.g., stock prices, economic indicators)."""
       
       # Generate correlated economic indicators
       np.random.seed(999)
       
       # Base economic trends
       trend1 = np.cumsum(np.random.randn(n_days) * 0.01)  # Random walk trend
       trend2 = np.cumsum(np.random.randn(n_days) * 0.01)
       
       # Economic cycles (business cycles, seasonal effects)
       t = np.arange(n_days)
       cycle1 = 0.1 * np.sin(2 * np.pi * t / 252) + 0.05 * np.sin(2 * np.pi * t / 365)  # Annual cycle
       cycle2 = 0.08 * np.sin(2 * np.pi * t / 180) + 0.06 * np.sin(2 * np.pi * t / 30)   # Quarterly cycle
       
       # Market volatility (GARCH-like)
       volatility = np.zeros(n_days)
       volatility[0] = 0.02
       
       for i in range(1, n_days):
           volatility[i] = 0.01 + 0.1 * (np.random.randn(1)[0]**2) + 0.8 * volatility[i-1]
       
       # Generate multiple trials (different market conditions)
       n_trials = 15  # Different economic scenarios
       data = np.zeros((n_series, n_days, n_trials))
       
       for trial in range(n_trials):
           # Economic shocks and events
           shock_times = np.random.choice(n_days, size=np.random.poisson(5), replace=False)
           shock_magnitudes = np.random.randn(len(shock_times)) * 0.05
           
           # Series 1: Leading economic indicator
           noise1 = np.random.randn(n_days) * volatility * (0.8 + 0.4 * np.random.randn())
           data[0, :, trial] = trend1 + cycle1 + noise1
           
           # Add economic shocks
           for shock_time, shock_mag in zip(shock_times, shock_magnitudes):
               data[0, shock_time:shock_time+10, trial] += shock_mag
           
           # Series 2: Lagging indicator (influenced by Series 1)
           noise2 = np.random.randn(n_days) * volatility * (0.9 + 0.2 * np.random.randn())
           
           for i in range(5, n_days):  # 5-day lag
               data[1, i, trial] = (
                   trend2[i] + cycle2[i] + noise2[i] +
                   0.3 * data[0, i-5, trial] +    # 5-day lagged influence
                   0.2 * data[0, i-3, trial] +    # 3-day lagged influence
                   0.1 * data[0, i-1, trial]      # 1-day lagged influence
               )
       
       return data
   
   # Analyze economic data
   print("\nAnalyzing simulated economic time series...")
   
   economic_data = simulate_economic_data(n_days=500, n_series=2)
   print(f"Economic data shape: {economic_data.shape}")
   
   # Difference the data to remove trends (common in economic analysis)
   diff_data = np.diff(economic_data, axis=1)  # First difference
   print(f"Differenced data shape: {diff_data.shape}")
   
   # Economic DCS analysis
   econ_calculator = DCSCalculator(
       model_order=8,             # Higher order for economic lags
       time_mode="inhomo",        # Non-stationary economic conditions
       use_diagonal_covariance=False
   )
   
   try:
       econ_result = econ_calculator.analyze(diff_data)
       
       print("Economic DCS Analysis Results:")
       print(f"Leading→Lagging causality: {econ_result.causal_strength[:, 1].mean():.4f}")
       print(f"Lagging→Leading causality: {econ_result.causal_strength[:, 0].mean():.4f}")
       
       # Expected: Leading should have stronger influence on Lagging
       if econ_result.causal_strength[:, 1].mean() > econ_result.causal_strength[:, 0].mean():
           print("✓ Expected pattern detected: Leading indicator influences lagging indicator")
       else:
           print("⚠ Unexpected pattern: Check data generation or model parameters")
       
       # Time-varying causality analysis
       causality_strength = econ_result.causal_strength[:, 1]  # Leading→Lagging
       
       # Identify periods of strong/weak causality
       strong_periods = causality_strength > np.percentile(causality_strength, 75)
       weak_periods = causality_strength < np.percentile(causality_strength, 25)
       
       print(f"Strong causality periods: {np.sum(strong_periods)} time points")
       print(f"Weak causality periods: {np.sum(weak_periods)} time points")
       
   except Exception as e:
       print(f"Economic analysis failed: {e}")

Multi-Scale Analysis
====================

.. code-block:: python

   # Multi-scale temporal analysis
   def multiscale_analysis(data, scales=[1, 2, 4, 8]):
       """Perform DCS analysis at multiple temporal scales."""
       
       results = {}
       
       for scale in scales:
           print(f"Analyzing at scale {scale}...")
           
           # Downsample data
           if scale == 1:
               scaled_data = data
           else:
               # Average over non-overlapping windows
               n_vars, n_time, n_trials = data.shape
               n_time_scaled = n_time // scale
               
               scaled_data = np.zeros((n_vars, n_time_scaled, n_trials))
               for i in range(n_time_scaled):
                   start_idx = i * scale
                   end_idx = start_idx + scale
                   scaled_data[:, i, :] = data[:, start_idx:end_idx, :].mean(axis=1)
           
           # DCS analysis at this scale
           try:
               # Adjust model order for scale
               model_order = max(2, 6 // scale)  # Fewer lags for coarser scales
               
               calc = DCSCalculator(
                   model_order=model_order,
                   time_mode="inhomo"
               )
               
               result = calc.analyze(scaled_data)
               
               results[scale] = {
                   'causal_strength': result.causal_strength,
                   'transfer_entropy': result.transfer_entropy,
                   'mean_dcs_xy': result.causal_strength[:, 1].mean(),
                   'mean_dcs_yx': result.causal_strength[:, 0].mean(),
                   'model_order': model_order
               }
               
               print(f"  Scale {scale}: DCS X→Y = {results[scale]['mean_dcs_xy']:.4f}")
               
           except Exception as e:
               print(f"  Scale {scale} failed: {e}")
               results[scale] = None
       
       return results
   
   # Perform multi-scale analysis on neural data
   print("\nMulti-scale causality analysis:")
   
   if 'neural_data_norm' in locals():
       multiscale_results = multiscale_analysis(neural_data_norm, scales=[1, 2, 4, 8])
       
       # Visualize scale-dependent causality
       scales = []
       causality_values = []
       
       for scale, result in multiscale_results.items():
           if result is not None:
               scales.append(scale)
               causality_values.append(result['mean_dcs_xy'])
       
       if len(scales) > 0:
           plt.figure(figsize=(10, 6))
           plt.semilogx(scales, causality_values, 'o-', linewidth=2, markersize=8)
           plt.xlabel('Temporal Scale')
           plt.ylabel('Mean Causal Strength (X→Y)')
           plt.title('Multi-Scale Causality Analysis')
           plt.grid(True, alpha=0.3)
           plt.show()
           
           print(f"Scale dependency analysis:")
           print(f"Fine scale (1): {causality_values[0]:.4f}")
           if len(causality_values) > 1:
               print(f"Coarse scale ({scales[-1]}): {causality_values[-1]:.4f}")
               
               if causality_values[0] > causality_values[-1]:
                   print("→ Causality stronger at fine temporal scales")
               else:
                   print("→ Causality stronger at coarse temporal scales")

***********************
Tutorial 5: Best Practices and Troubleshooting
***********************

Learn best practices for robust DCS analysis and how to troubleshoot common issues.

Data Quality Assessment
=======================

.. code-block:: python

   from trancit.utils.preprocess import check_data_quality, normalize_data
   import warnings
   
   def comprehensive_data_check(data, description=""):
       """Perform comprehensive data quality assessment."""
       
       print(f"\n=== Data Quality Assessment: {description} ===")
       
       # Basic properties
       print(f"Shape: {data.shape}")
       print(f"Data type: {data.dtype}")
       print(f"Memory usage: {data.nbytes / 1024**2:.2f} MB")
       
       # Statistical properties
       print(f"Mean: {data.mean():.4f}")
       print(f"Std: {data.std():.4f}")
       print(f"Min: {data.min():.4f}")
       print(f"Max: {data.max():.4f}")
       
       # Check for problematic values
       n_nan = np.isnan(data).sum()
       n_inf = np.isinf(data).sum()
       n_zero = (data == 0).sum()
       
       print(f"NaN values: {n_nan}")
       print(f"Inf values: {n_inf}")
       print(f"Zero values: {n_zero}")
       
       if n_nan > 0:
           warnings.warn(f"Found {n_nan} NaN values - may cause analysis failure")
       if n_inf > 0:
           warnings.warn(f"Found {n_inf} infinite values - may cause numerical issues")
       
       # Check variance across trials
       if data.ndim == 3:
           trial_vars = np.var(data, axis=(0, 1))  # Variance of each trial
           low_var_trials = np.sum(trial_vars < 0.01 * np.median(trial_vars))
           
           print(f"Low variance trials: {low_var_trials}/{data.shape[2]}")
           if low_var_trials > data.shape[2] * 0.2:
               warnings.warn("Many trials have very low variance - check data quality")
       
       # Check stationarity (simplified)
       if data.ndim >= 2:
           first_half_mean = data[:, :data.shape[1]//2].mean()
           second_half_mean = data[:, data.shape[1]//2:].mean()
           mean_diff = abs(first_half_mean - second_half_mean)
           
           print(f"First/second half mean difference: {mean_diff:.4f}")
           if mean_diff > data.std():
               print("⚠ Large mean differences between halves - data may be non-stationary")
           else:
               print("✓ Mean appears relatively stable")
       
       # Recommended actions
       recommendations = []
       
       if n_nan > 0 or n_inf > 0:
           recommendations.append("Remove or interpolate NaN/Inf values")
       
       if data.std() < 1e-6:
           recommendations.append("Data has very low variance - check scaling")
       elif data.std() > 1e6:
           recommendations.append("Data has very high variance - consider normalization")
       
       if data.shape[1] < 50:
           recommendations.append("Short time series - consider lower model order")
       
       if data.ndim == 3 and data.shape[2] < 5:
           recommendations.append("Few trials - results may be less robust")
       
       if recommendations:
           print("Recommendations:")
           for rec in recommendations:
               print(f"  • {rec}")
       else:
           print("✓ Data quality looks good")
       
       return {
           'n_nan': n_nan,
           'n_inf': n_inf,
           'mean_diff': mean_diff if data.ndim >= 2 else None,
           'recommendations': recommendations
       }
   
   # Test with various data quality scenarios
   print("Testing data quality assessment...")
   
   # 1. Good quality data
   good_data, _, _ = generate_signals(T=400, Ntrial=20, h=0.1,
                                     gamma1=0.5, gamma2=0.5,
                                     Omega1=1.0, Omega2=1.2)
   comprehensive_data_check(good_data, "Good Quality Data")
   
   # 2. Problematic data
   bad_data = good_data.copy()
   bad_data[0, 100:110, :] = np.nan  # Introduce NaN values
   bad_data[1, 200, 0] = np.inf      # Introduce Inf value
   bad_data *= 1e8                   # Make values very large
   
   comprehensive_data_check(bad_data, "Problematic Data")

Robust Analysis Pipeline
========================

.. code-block:: python

   def robust_dcs_analysis(data, description="", max_model_order=8):
       """Perform robust DCS analysis with automatic parameter adjustment."""
       
       print(f"\n=== Robust DCS Analysis: {description} ===")
       
       # Step 1: Data quality check
       quality_results = comprehensive_data_check(data, description)
       
       # Step 2: Data preprocessing based on quality assessment
       processed_data = data.copy()
       
       if quality_results['n_nan'] > 0 or quality_results['n_inf'] > 0:
           print("Cleaning data...")
           # Replace NaN and Inf with interpolated values
           from scipy.interpolate import interp1d
           
           for var in range(processed_data.shape[0]):
               for trial in range(processed_data.shape[2]):
                   signal = processed_data[var, :, trial]
                   
                   # Find valid (non-NaN, non-Inf) indices
                   valid_mask = np.isfinite(signal)
                   
                   if np.sum(valid_mask) > 10:  # Need some valid points
                       valid_indices = np.where(valid_mask)[0]
                       invalid_indices = np.where(~valid_mask)[0]
                       
                       if len(invalid_indices) > 0:
                           # Linear interpolation
                           f = interp1d(valid_indices, signal[valid_indices], 
                                       bounds_error=False, fill_value='extrapolate')
                           processed_data[var, invalid_indices, trial] = f(invalid_indices)
       
       # Normalize data
       if processed_data.std() > 1000 or processed_data.std() < 0.001:
           print("Normalizing data...")
           processed_data = normalize_data(processed_data, method="zscore", axis=1)
       
       # Step 3: Adaptive model order selection
       n_time = processed_data.shape[1]
       max_reasonable_order = min(max_model_order, n_time // 10)  # Rule of thumb
       
       print(f"Testing model orders from 1 to {max_reasonable_order}...")
       
       best_order = None
       best_result = None
       order_scores = {}
       
       for order in range(1, max_reasonable_order + 1):
           try:
               calc = DCSCalculator(model_order=order, time_mode="inhomo")
               result = calc.analyze(processed_data)
               
               # Score based on finite values and reasonable magnitudes
               dcs_values = result.causal_strength
               
               if np.all(np.isfinite(dcs_values)) and np.all(dcs_values >= 0):
                   # Simple scoring: prefer moderate values, penalize extreme values
                   mean_dcs = dcs_values.mean()
                   std_dcs = dcs_values.std()
                   
                   score = mean_dcs - 2 * (std_dcs > mean_dcs * 2)  # Penalize high variability
                   order_scores[order] = score
                   
                   if best_order is None or score > order_scores[best_order]:
                       best_order = order
                       best_result = result
                   
                   print(f"  Order {order}: Score = {score:.4f}")
               else:
                   print(f"  Order {order}: Failed (non-finite or negative values)")
           
           except Exception as e:
               print(f"  Order {order}: Failed ({str(e)[:50]}...)")
       
       # Step 4: Final analysis with best parameters
       if best_result is not None:
           print(f"\nBest model order: {best_order}")
           print(f"Final results:")
           print(f"  DCS X→Y: {best_result.causal_strength[:, 1].mean():.4f}")
           print(f"  DCS Y→X: {best_result.causal_strength[:, 0].mean():.4f}")
           print(f"  TE X→Y: {best_result.transfer_entropy[:, 1].mean():.4f}")
           print(f"  TE Y→X: {best_result.transfer_entropy[:, 0].mean():.4f}")
           
           # Confidence assessment
           dcs_xy_std = best_result.causal_strength[:, 1].std()
           dcs_yx_std = best_result.causal_strength[:, 0].std()
           
           print(f"  DCS X→Y variability: {dcs_xy_std:.4f}")
           print(f"  DCS Y→X variability: {dcs_yx_std:.4f}")
           
           if dcs_xy_std < 0.1 and dcs_yx_std < 0.1:
               print("  ✓ Low variability - results appear stable")
           else:
               print("  ⚠ High variability - results may be less reliable")
           
           return best_result, best_order, processed_data
       
       else:
           print("❌ No successful analysis found")
           print("Recommendations:")
           print("  • Check data format (should be 3D: variables × time × trials)")
           print("  • Ensure sufficient data length (>100 time points recommended)")
           print("  • Verify data contains meaningful signal (not just noise)")
           return None, None, processed_data
   
   # Test robust analysis
   test_data, _, _ = generate_signals(T=300, Ntrial=15, h=0.1,
                                     gamma1=0.4, gamma2=0.6,
                                     Omega1=0.9, Omega2=1.3)
   
   robust_result, best_order, clean_data = robust_dcs_analysis(test_data, "Test Analysis")

Common Issues and Solutions
===========================

.. code-block:: python

   # Demonstrate common issues and their solutions
   
   print("\n=== Common Issues and Solutions ===")
   
   # Issue 1: Insufficient data length
   print("\n1. Issue: Insufficient data length")
   short_data, _, _ = generate_signals(T=50, Ntrial=10, h=0.1,  # Very short
                                      gamma1=0.5, gamma2=0.5,
                                      Omega1=1.0, Omega2=1.2)
   
   print("Attempting analysis with very short data...")
   try:
       calc = DCSCalculator(model_order=10, time_mode="inhomo")  # Too high order
       result = calc.analyze(short_data)
       print("Analysis succeeded (unexpected)")
   except Exception as e:
       print(f"Analysis failed as expected: {type(e).__name__}")
       print("Solution: Use lower model order or collect more data")
       
       # Solution
       try:
           calc_fixed = DCSCalculator(model_order=2, time_mode="homo")  # Lower order
           result_fixed = calc_fixed.analyze(short_data)
           print(f"✓ Fixed analysis succeeded: DCS X→Y = {result_fixed.causal_strength[:, 1].mean():.4f}")
       except Exception as e2:
           print(f"Still failed: {e2}")
   
   # Issue 2: Highly correlated/singular data
   print("\n2. Issue: Highly correlated data")
   T, N = 200, 15
   
   # Create perfectly correlated data
   base_signal = np.random.randn(T, N)
   singular_data = np.zeros((2, T, N))
   singular_data[0] = base_signal
   singular_data[1] = base_signal + 1e-10 * np.random.randn(T, N)  # Nearly identical
   
   print("Attempting analysis with highly correlated data...")
   try:
       calc = DCSCalculator(model_order=4, time_mode="inhomo")
       result = calc.analyze(singular_data)
       print("Analysis succeeded (unexpected)")
   except Exception as e:
       print(f"Analysis failed as expected: {type(e).__name__}")
       print("Solution: Use diagonal covariance approximation or add regularization")
       
       # Solution
       try:
           calc_fixed = DCSCalculator(model_order=3, 
                                     time_mode="inhomo",
                                     use_diagonal_covariance=True)  # Diagonal approximation
           result_fixed = calc_fixed.analyze(singular_data)
           print(f"✓ Fixed analysis succeeded: DCS X→Y = {result_fixed.causal_strength[:, 1].mean():.4f}")
       except Exception as e2:
           print(f"Still failed: {e2}")
   
   # Issue 3: No events detected in pipeline
   print("\n3. Issue: No events detected in pipeline")
   
   # Create very smooth signals (no clear events)
   smooth_data, _, _ = generate_signals(T=500, Ntrial=10, h=0.1,
                                       gamma1=0.1, gamma2=0.1,  # Low damping = smooth
                                       Omega1=1.0, Omega2=1.0)
   
   smooth_signal = np.mean(smooth_data, axis=2)
   
   # Try with high threshold (likely to fail)
   from trancit.config import PipelineConfig, PipelineOptions, DetectionParams, CausalParams, BicParams, OutputParams
   
   high_threshold_config = PipelineConfig(
       options=PipelineOptions(detection=True, causal_analysis=True),
       detection=DetectionParams(thres_ratio=5.0, l_extract=50, l_start=25),  # Very high threshold
       causal=CausalParams(ref_time=25, estim_mode="OLS"),
       bic=BicParams(),
       output=OutputParams()
   )
   
   print("Attempting pipeline with high detection threshold...")
   try:
       orchestrator = PipelineOrchestrator(high_threshold_config)
       result = orchestrator.run(smooth_signal, smooth_signal * 1.1)
       
       if result.results.get('locs') is not None:
           n_events = len(result.results['locs'])
           print(f"Found {n_events} events")
       else:
           print("No events detected")
       
   except Exception as e:
       print(f"Pipeline failed: {type(e).__name__}")
   
   print("Solution: Lower detection threshold")
   
   # Solution: Lower threshold
   low_threshold_config = PipelineConfig(
       options=PipelineOptions(detection=True, causal_analysis=True),
       detection=DetectionParams(thres_ratio=1.5, l_extract=50, l_start=25),  # Lower threshold
       causal=CausalParams(ref_time=25, estim_mode="OLS"),
       bic=BicParams(),
       output=OutputParams()
   )
   
   try:
       orchestrator_fixed = PipelineOrchestrator(low_threshold_config)
       result_fixed = orchestrator_fixed.run(smooth_signal, smooth_signal * 1.5)  # More amplification
       
       if result_fixed.results.get('locs') is not None:
           n_events = len(result_fixed.results['locs'])
           print(f"✓ Fixed pipeline found {n_events} events")
       else:
           print("Still no events detected - signal may be too smooth")
           
   except Exception as e:
       print(f"Fixed pipeline still failed: {e}")

Performance Monitoring
======================

.. code-block:: python

   import time
   import psutil
   import os
   
   def monitor_analysis_performance(data, description="", verbose=True):
       """Monitor memory and time performance of DCS analysis."""
       
       if verbose:
           print(f"\n=== Performance Monitoring: {description} ===")
       
       # Initial system state
       process = psutil.Process(os.getpid())
       initial_memory = process.memory_info().rss / 1024**2  # MB
       start_time = time.time()
       
       if verbose:
           print(f"Initial memory usage: {initial_memory:.1f} MB")
           print(f"Data size: {data.nbytes / 1024**2:.2f} MB")
       
       try:
           # Perform analysis
           calc = DCSCalculator(model_order=4, time_mode="inhomo")
           result = calc.analyze(data)
           
           # Final system state
           end_time = time.time()
           final_memory = process.memory_info().rss / 1024**2
           
           analysis_time = end_time - start_time
           memory_increase = final_memory - initial_memory
           
           if verbose:
               print(f"Analysis time: {analysis_time:.2f} seconds")
               print(f"Memory increase: {memory_increase:.1f} MB")
               print(f"Peak memory usage: {final_memory:.1f} MB")
               
               # Performance metrics
               data_throughput = data.size / analysis_time  # elements per second
               print(f"Data throughput: {data_throughput/1000:.1f}K elements/second")
               
               # Efficiency assessment
               if analysis_time < 1.0:
                   print("✓ Fast analysis")
               elif analysis_time < 10.0:
                   print("○ Moderate analysis time")
               else:
                   print("⚠ Slow analysis - consider optimization")
               
               if memory_increase < 100:
                   print("✓ Low memory overhead")
               elif memory_increase < 500:
                   print("○ Moderate memory usage")
               else:
                   print("⚠ High memory usage - consider processing in chunks")
           
           return {
               'analysis_time': analysis_time,
               'memory_increase': memory_increase,
               'data_throughput': data_throughput,
               'success': True
           }
           
       except Exception as e:
           end_time = time.time()
           analysis_time = end_time - start_time
           
           if verbose:
               print(f"Analysis failed after {analysis_time:.2f} seconds: {e}")
           
           return {
               'analysis_time': analysis_time,
               'memory_increase': 0,
               'data_throughput': 0,
               'success': False,
               'error': str(e)
           }
   
   # Test performance with different data sizes
   data_sizes = [
       (100, 10, "Small"),
       (500, 20, "Medium"), 
       (1000, 30, "Large")
   ]
   
   performance_results = {}
   
   for T, N, size_name in data_sizes:
       test_data, _, _ = generate_signals(T=T, Ntrial=N, h=0.1,
                                         gamma1=0.5, gamma2=0.5,
                                         Omega1=1.0, Omega2=1.2)
       
       perf_result = monitor_analysis_performance(test_data, f"{size_name} Dataset ({T}×{N})")
       performance_results[size_name] = perf_result
   
   # Performance summary
   print("\n=== Performance Summary ===")
   for size_name, result in performance_results.items():
       if result['success']:
           print(f"{size_name}: {result['analysis_time']:.2f}s, {result['memory_increase']:.1f}MB")
       else:
           print(f"{size_name}: Failed - {result['error']}")
