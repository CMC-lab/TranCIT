.. _quickstart:

##########
Quickstart
##########

This quickstart guide will get you up and running with **TranCIT: Transient Causal Interaction** analysis in just a few minutes. 

We'll cover the most common use cases with the new class-based API, which provides cleaner interfaces and better error handling than the previous function-based approach.

******************
Installation Check
******************

First, verify your installation:

.. code-block:: python

   import trancit
   print(f"TranCIT version: {trancit.__version__}")

If this works, you're ready to go! If not, see the :doc:`installation` guide.

**************************
Basic Causality Analysis
**************************

The simplest way to analyze causal relationships is using the ``DCSCalculator`` class:

.. code-block:: python
   :linenos:

   import numpy as np
   from trancit import DCSCalculator, generate_signals

   # Generate synthetic bivariate time series data
   data, _, _ = generate_signals(
       T=1000, Ntrial=20, h=0.1, 
       gamma1=0.5, gamma2=0.5, 
       Omega1=1.0, Omega2=1.2
   )
   
   print(f"Data shape: {data.shape}")  # (2, 1000, 20) = (variables, time, trials)

   # Create DCS calculator
   calculator = DCSCalculator(model_order=4, time_mode="inhomo")
   
   # Perform analysis
   result = calculator.analyze(data)
   
   # Access results
   print(f"Causal Strength shape: {result.causal_strength.shape}")
   print(f"Transfer Entropy shape: {result.transfer_entropy.shape}")
   print(f"Granger Causality shape: {result.granger_causality.shape}")
   
   # Display mean causality values
   print(f"Mean DCS (X→Y): {result.causal_strength[:, 0].mean():.4f}")
   print(f"Mean DCS (Y→X): {result.causal_strength[:, 1].mean():.4f}")

**Key Points:**

- ``model_order=4``: Number of time lags to include in the VAR model
- ``time_mode="inhomo"``: Use time-inhomogeneous analysis (recommended)
- Results contain causality measures with shape ``(n_time_points, 2)`` where column 0 is Y→X and column 1 is X→Y

*****************************
Pipeline-Based Event Analysis
*****************************

For event-based analysis (detecting and analyzing specific time windows), use the ``PipelineOrchestrator``:

.. code-block:: python
   :linenos:

   import numpy as np
   from trancit import generate_signals, PipelineOrchestrator
   from trancit.config import (
       PipelineConfig, PipelineOptions, DetectionParams, 
       CausalParams, BicParams, OutputParams
   )

   # Generate synthetic data
   data, _, _ = generate_signals(
       T=1200, Ntrial=20, h=0.1,
       gamma1=0.5, gamma2=0.5, 
       Omega1=1.0, Omega2=1.2
   )
   
   # Prepare signals for pipeline
   original_signal = np.mean(data, axis=2)  # Average across trials
   detection_signal = original_signal * 1.5  # Amplify for event detection
   
   # Configure the analysis pipeline
   config = PipelineConfig(
       options=PipelineOptions(
           detection=True,           # Enable event detection
           bic=False,               # Skip BIC model selection (faster)
           causal_analysis=True,    # Enable causality analysis
           bootstrap=False,         # Skip bootstrap (faster)
           save_flag=False,         # Don't save intermediate results
           debiased_stats=False     # Skip debiased analysis
       ),
       detection=DetectionParams(
           thres_ratio=2.0,         # Detection threshold (higher = fewer events)
           align_type="peak",       # Align events to peaks
           l_extract=150,           # Length of extracted windows
           l_start=75,              # Start offset within windows
           remove_artif=True        # Remove artifact-contaminated trials
       ),
       causal=CausalParams(
           ref_time=75,             # Reference time for rDCS calculation
           estim_mode="OLS"         # Ordinary Least Squares estimation
       ),
       bic=BicParams(morder=4),
       output=OutputParams(file_keyword="quickstart_example")
   )
   
   # Run the analysis pipeline
   orchestrator = PipelineOrchestrator(config)
   
   try:
       result = orchestrator.run(original_signal, detection_signal)
       
       # Access results
       if result.results.get("CausalOutput"):
           causal_output = result.results["CausalOutput"]["OLS"]
           
           if "DCS" in causal_output:
               dcs_values = causal_output["DCS"]
               print(f"DCS shape: {dcs_values.shape}")
               print(f"Number of events detected: {dcs_values.shape[0]}")
               print(f"Mean DCS (X→Y): {dcs_values[:, 1].mean():.4f}")
               print(f"Mean DCS (Y→X): {dcs_values[:, 0].mean():.4f}")
           
           if "TE" in causal_output:
               te_values = causal_output["TE"]
               print(f"Mean TE (X→Y): {te_values[:, 1].mean():.4f}")
               print(f"Mean TE (Y→X): {te_values[:, 0].mean():.4f}")
               
           if "rDCS" in causal_output:
               rdcs_values = causal_output["rDCS"]
               print(f"Mean rDCS (X→Y): {rdcs_values[:, 1].mean():.4f}")
               print(f"Mean rDCS (Y→X): {rdcs_values[:, 0].mean():.4f}")
       else:
           print("No causal output generated - check signal characteristics")
           
   except Exception as e:
       print(f"Pipeline analysis failed: {e}")
       print("Tip: Try adjusting thres_ratio or using simpler configuration")

**Key Pipeline Components:**

- **Event Detection**: Finds time windows of interest based on signal characteristics
- **Snapshot Extraction**: Extracts fixed-length windows around detected events
- **Causality Analysis**: Computes DCS, TE, and rDCS for each event window
- **Bootstrap Analysis**: Optional statistical significance testing

*********************************
Different Types of Analysis
*********************************

DCS provides several specialized calculators for different causality measures:

.. code-block:: python
   :linenos:

   from trancit import (
       DCSCalculator,              # Dynamic Causal Strength
       TransferEntropyCalculator,  # Information-theoretic measure
       GrangerCausalityCalculator, # Linear causality detection
       RelativeDCSCalculator       # Event-based relative causality
   )
   
   # Sample bivariate data
   data = np.random.randn(2, 500, 15)
   
   # 1. Dynamic Causal Strength Analysis
   dcs_calc = DCSCalculator(model_order=3, time_mode="inhomo")
   dcs_result = dcs_calc.analyze(data)
   print(f"DCS computed for {dcs_result.causal_strength.shape[0]} time points")
   
   # 2. Transfer Entropy Analysis  
   te_calc = TransferEntropyCalculator(model_order=3)
   te_result = te_calc.analyze(data)
   print(f"TE computed: {te_result.transfer_entropy.shape}")
   
   # 3. Granger Causality Analysis
   gc_calc = GrangerCausalityCalculator(model_order=3)
   gc_result = gc_calc.analyze(data)
   print(f"GC p-values shape: {gc_result.pvalues.shape}")
   
   # 4. Relative DCS (requires event data and statistics)
   # This is typically used within the pipeline, but can be used standalone
   # rdcs_calc = RelativeDCSCalculator(model_order=3, reference_time=25)
   # rdcs_result = rdcs_calc.analyze(event_data, event_stats)

*****************************
Working with Real Data
*****************************

Here's how to apply DCS to your own time series data:

.. code-block:: python
   :linenos:

   import numpy as np
   from trancit import DCSCalculator
   from trancit.utils.preprocess import normalize_data
   
   # Load your data (example with NumPy)
   # Your data should have shape (n_variables, n_timepoints, n_trials)
   # For DCS: n_variables must be 2 (bivariate analysis)
   
   # Example: loading data from a file
   # data = np.load('my_time_series.npy')  # Shape should be (2, T, N)
   
   # For demonstration, create sample data
   np.random.seed(42)
   
   # Simulate two coupled time series
   T, N = 1000, 25  # 1000 time points, 25 trials
   
   # Generate correlated signals (simple example)
   noise1 = np.random.randn(T, N)
   noise2 = np.random.randn(T, N)
   
   signal1 = np.zeros((T, N))
   signal2 = np.zeros((T, N))
   
   # Create coupling: X influences Y with delay
   for t in range(3, T):
       signal1[t] = 0.7 * signal1[t-1] - 0.1 * signal1[t-2] + noise1[t]
       signal2[t] = 0.6 * signal2[t-1] + 0.3 * signal1[t-3] + noise2[t]  # Y depends on past X
   
   # Arrange in DCS format: (n_vars, n_time, n_trials)
   data = np.array([signal1.T, signal2.T])
   
   print(f"Data shape: {data.shape}")
   
   # Optional: normalize your data
   data_normalized = normalize_data(data, method="zscore", axis=1)
   
   # Perform DCS analysis
   calculator = DCSCalculator(model_order=5, time_mode="inhomo")
   
   try:
       result = calculator.analyze(data_normalized)
       
       print("Analysis successful!")
       print(f"X→Y causality: {result.causal_strength[:, 1].mean():.4f}")
       print(f"Y→X causality: {result.causal_strength[:, 0].mean():.4f}")
       
       # Plot results (optional)
       try:
           import matplotlib.pyplot as plt
           
           plt.figure(figsize=(12, 4))
           plt.subplot(1, 2, 1)
           plt.plot(result.causal_strength[:, 1], label='X→Y', alpha=0.7)
           plt.plot(result.causal_strength[:, 0], label='Y→X', alpha=0.7) 
           plt.xlabel('Time')
           plt.ylabel('Causal Strength')
           plt.legend()
           plt.title('Dynamic Causal Strength')
           
           plt.subplot(1, 2, 2)
           plt.plot(result.transfer_entropy[:, 1], label='X→Y', alpha=0.7)
           plt.plot(result.transfer_entropy[:, 0], label='Y→X', alpha=0.7)
           plt.xlabel('Time')
           plt.ylabel('Transfer Entropy') 
           plt.legend()
           plt.title('Transfer Entropy')
           
           plt.tight_layout()
           plt.show()
           
       except ImportError:
           print("Install matplotlib to plot results: pip install matplotlib")
       
   except Exception as e:
       print(f"Analysis failed: {e}")
       print("Check your data format and try adjusting model_order")

**************************
Common Troubleshooting
**************************

**Issue: "Input data must be bivariate"**

- DCS is designed for analyzing relationships between two time series
- Ensure your data has shape ``(2, n_timepoints, n_trials)``

**Issue: "Insufficient observations"**

- Your time series is too short for the chosen model order
- Try reducing ``model_order`` or collecting more data
- Rule of thumb: need at least ``model_order * 10`` time points

**Issue: "Singular matrix errors"**

- Your data may have perfect correlations or insufficient variation
- Try normalizing your data: ``normalize_data(data, method="zscore")``
- Consider adding small amount of noise for numerical stability

**Issue: Pipeline finds no events**

- Your ``thres_ratio`` might be too high - try lowering it (e.g., 1.5 instead of 3.0)
- Check if your detection signal has sufficient variability
- Visualize your signal to understand its characteristics

**************
What's Next?
**************

Now that you understand the basics:

- **Explore Examples**: Check out ``examples/basic_usage.py`` and ``examples/lfp_pipeline.py``
- **Read API Documentation**: See :doc:`api` for complete class and function references
- **Learn Advanced Features**: Check :doc:`examples` for specialized use cases
- **Understand the Science**: Read about the theoretical background in our `scientific paper <https://www.frontiersin.org/journals/network-physiology/articles/10.3389/fnetp.2023.1085347/full>`_

**Configuration Tips:**

- Start with ``model_order=3`` or ``4`` for most applications
- Use ``time_mode="inhomo"`` for non-stationary signals (recommended)
- Enable ``bootstrap=True`` in pipeline for statistical significance testing
- Use ``BIC`` model selection for automatic model order selection

Need help? Check our :doc:`../TROUBLESHOOTING` guide or open an issue on `GitHub <https://github.com/CMC-lab/TranCIT/issues>`_.
