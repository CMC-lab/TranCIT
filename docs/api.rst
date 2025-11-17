.. _api_reference:

############
API Reference
############

Welcome to the comprehensive API reference for **TranCIT: Transient Causal Interaction Toolbox**. This section provides detailed documentation for all classes, functions, and modules in the package.

The TranCIT package is organized into several core components:

- **Core Classes**: Base analyzers, results, and configuration objects
- **Causality Analysis**: Dynamic Causal Strength (DCS), relative Dynamic Causal Strength (rDCS), Transfer Entropy (TE), Granger Causality (GC) calculators
- **Pipeline System**: Event-based analysis orchestration and stages
- **Model Estimation**: VAR models, BIC selection, and validation
- **Simulation**: Synthetic data generation for testing and validation
- **Utilities**: Data preprocessing, visualization, and helper functions

.. contents:: API Contents
   :local:
   :depth: 3

****************************
Core Module (`trancit.core`)
****************************

The core module provides the foundational classes and interfaces used throughout the TranCIT package.

Base Classes
============

.. currentmodule:: trancit.core.base

.. autoclass:: BaseAnalyzer
   :members:
   :show-inheritance:
   
   The abstract base class for all analysis components. All calculators and analyzers inherit from this class, providing a consistent interface.
   
   **Key Methods:**
   
   - ``analyze(data, **kwargs)``: Main analysis method (must be implemented by subclasses)
   - ``validate_config(config)``: Validates configuration parameters
   - ``_log_analysis_start()``, ``_log_analysis_complete()``: Logging support
   
   **Example:**
   
   .. code-block:: python
   
      from trancit.core.base import BaseAnalyzer
      import numpy as np
      
      class MyAnalyzer(BaseAnalyzer):
          def analyze(self, data):
              # Your analysis implementation here
              return MyResult(result=data.sum())

.. autoclass:: BaseResult
   :members:
   :show-inheritance:
   
   Base class for all result objects. Provides common functionality for storing and accessing analysis results.
   
   **Key Methods:**
   
   - ``to_dict()``: Convert result to dictionary format
   - ``__getitem__(key)``: Dictionary-like access to results
   
   **Example:**
   
   .. code-block:: python
   
      from trancit.core.base import BaseResult
      
      result = MyResult(causal_strength=cs_array, transfer_entropy=te_array)
      print(result.causal_strength.shape)
      result_dict = result.to_dict()

.. autoclass:: BaseConfig
   :members:
   :show-inheritance:
   
   Base class for configuration objects. Provides parameter validation and default value handling.

Exception Classes
=================

.. currentmodule:: trancit.core.exceptions

.. autoclass:: DCSError
   :show-inheritance:
   
   Base exception class for all DCS-related errors.

.. autoclass:: ValidationError
   :show-inheritance:
   
   Raised when input data or parameters fail validation checks.
   
   **Common Causes:**
   
   - Wrong data dimensions (should be 3D: variables × time × trials)
   - Non-bivariate data (DCS requires exactly 2 variables)
   - Model order too large for available data
   - Invalid parameter values

.. autoclass:: ComputationError
   :show-inheritance:
   
   Raised when numerical computation fails.
   
   **Common Causes:**
   
   - Singular or near-singular matrices
   - Numerical instability
   - Insufficient data for reliable estimation

.. autoclass:: ConfigurationError
   :show-inheritance:
   
   Raised when configuration parameters are invalid or inconsistent.

**********************************************
Causality Analysis (`trancit.causality`)
**********************************************

The causality module provides implementations of various causality measures.

Dynamic Causal Strength (DCS)
==============================

.. currentmodule:: trancit.causality.dcs

.. autoclass:: DCSCalculator
   :members:
   :show-inheritance:
   
   Main class for Dynamic Causal Strength analysis. Computes time-varying causal relationships using VAR models.
   
   **Parameters:**
   
   - ``model_order`` (int): Number of time lags to include (typically 2-8)
   - ``time_mode`` (str): "inhomo" (time-varying) or "homo" (time-invariant)
   - ``use_diagonal_covariance`` (bool): Use diagonal approximation for numerical stability
   
   **Key Methods:**
   
   - ``analyze(data)``: Perform DCS analysis on time series data
   - ``_validate_input_data(data)``: Check data format and quality
   - ``_prepare_data_structures(data)``: Prepare data for VAR analysis
   
   **Example:**
   
   .. code-block:: python
   
      from trancit import DCSCalculator
      import numpy as np
      
      # Generate sample data
      data = np.random.randn(2, 500, 20)  # 2 vars, 500 time points, 20 trials
      
      # Create calculator
      calculator = DCSCalculator(model_order=4, time_mode="inhomo")
      
      # Perform analysis
      result = calculator.analyze(data)
      
      print(f"Causal strength shape: {result.causal_strength.shape}")
      print(f"Mean X→Y causality: {result.causal_strength[:, 1].mean():.4f}")

.. autoclass:: DCSResult
   :members:
   :show-inheritance:
   
   Result object containing DCS analysis outputs.
   
   **Attributes:**
   
   - ``causal_strength`` (ndarray): Dynamic causal strength values, shape (time, 2)
   - ``transfer_entropy`` (ndarray): Transfer entropy values, shape (time, 2) 
   - ``granger_causality`` (ndarray): Granger causality values, shape (time, 2)
   - ``coefficients`` (ndarray): VAR model coefficients
   - ``te_residual_cov`` (ndarray): Residual covariance matrices
   
   **Column Convention:**
   
   - Column 0: Y → X (variable 2 influences variable 1)
   - Column 1: X → Y (variable 1 influences variable 2)

Transfer Entropy
================

.. currentmodule:: trancit.causality.transfer_entropy

.. autoclass:: TransferEntropyCalculator
   :members:
   :show-inheritance:
   
   Calculator for Transfer Entropy, an information-theoretic measure of causality.
   
   **About Transfer Entropy:**
   
   Transfer Entropy (TE) quantifies the information flow between time series. It measures how much knowing the past of one variable reduces uncertainty about the future of another variable.
   
   **Example:**
   
   .. code-block:: python
   
      from trancit.causality import TransferEntropyCalculator
      import numpy as np
      
      # Generate sample data
      data = np.random.randn(2, 500, 20)  # 2 vars, 500 time points, 20 trials
      
      calculator = TransferEntropyCalculator(model_order=3)
      result = calculator.analyze(data)
      
      print(f"TE X→Y: {result.transfer_entropy[:, 1].mean():.4f}")
      print(f"TE Y→X: {result.transfer_entropy[:, 0].mean():.4f}")

.. autoclass:: TransferEntropyResult
   :members:
   :show-inheritance:

Granger Causality
=================

.. currentmodule:: trancit.causality.granger

.. autoclass:: GrangerCausalityCalculator
   :members:
   :show-inheritance:
   
   Calculator for Granger Causality, a linear measure of predictive causality.
   
   **About Granger Causality:**
   
   Granger Causality tests whether past values of one time series help predict future values of another, beyond what can be predicted from the target series alone.
   
   **Example:**
   
   .. code-block:: python
   
      from trancit.causality import GrangerCausalityCalculator
      import numpy as np
      
      # Generate sample data
      data = np.random.randn(2, 500, 20)  # 2 vars, 500 time points, 20 trials
      
      calculator = GrangerCausalityCalculator(model_order=4)
      result = calculator.analyze(data)
      
      # Check significance
      significant_xy = result.pvalues[:, 1] < 0.05
      print(f"Significant X→Y causality: {significant_xy.sum()} time points")

.. autoclass:: GrangerCausalityResult
   :members:
   :show-inheritance:
   
   **Additional Attributes:**
   
   - ``pvalues`` (ndarray): Statistical significance p-values
   - ``f_statistics`` (ndarray): F-test statistics

Relative Dynamic Causal Strength (rDCS)
========================================

.. currentmodule:: trancit.causality.rdcs

.. autoclass:: RelativeDCSCalculator
   :members:
   :show-inheritance:
   
   Calculator for Relative Dynamic Causal Strength, used for event-based analysis.
   
   **About rDCS:**
   
   Relative DCS measures causal strength relative to a baseline reference time, making it particularly useful for analyzing event-related changes in causality.
   
   **Example:**
   
   .. code-block:: python
   
      from trancit.causality import RelativeDCSCalculator
      import numpy as np
      
      # Note: RelativeDCSCalculator requires event data and statistics
      # typically generated by the pipeline. This is a simplified example.
      # In practice, you would use PipelineOrchestrator to generate these.
      
      calculator = RelativeDCSCalculator(
          model_order=3,
          reference_time=50,  # Reference time index
          estimation_mode="OLS"
      )
      
      # Requires event data and statistics (typically from pipeline)
      # result = calculator.analyze(event_data, event_stats)

.. autoclass:: RelativeDCSResult
   :members:
   :show-inheritance:

Utility Functions
=================

.. currentmodule:: trancit.causality.rdcs

.. autofunction:: time_varying_causality

   Core function for computing time-varying causality measures. Used internally by the pipeline and can be called directly for advanced use cases.
   
   **Parameters:**
   
   - ``event_data`` (ndarray): Event data array, shape (nvar × (model_order + 1), nobs, ntrials)
   - ``stats`` (dict): Model statistics containing coefficients and covariances
   - ``causal_params`` (dict): Analysis parameters
   
   **Returns:**
   
   Dictionary containing 'TE', 'DCS', and 'rDCS' arrays.
   
   **Example:**
   
   .. code-block:: python
   
      from trancit.causality import time_varying_causality
      import numpy as np
      
      # Note: This function requires event data and statistics
      # typically generated by the pipeline. This is a simplified example.
      # In practice, you would use PipelineOrchestrator to generate these.
      
      # causal_params = {
      #     'ref_time': 25,
      #     'estim_mode': 'OLS',
      #     'morder': 3,
      #     'diag_flag': False,
      #     'old_version': False
      # }
      # 
      # causality_results = time_varying_causality(event_data, stats, causal_params)
      # 
      # dcs_values = causality_results['DCS']
      # te_values = causality_results['TE']
      # rdcs_values = causality_results['rDCS']

************************************
Pipeline System (`trancit.pipeline`)
************************************

The pipeline module provides orchestrated analysis workflows for event-based DCS analysis.

Pipeline Orchestration
=======================

.. currentmodule:: trancit.pipeline.orchestrator

.. autoclass:: PipelineOrchestrator
   :members:
   :show-inheritance:
   
   Main orchestrator class that coordinates the complete analysis pipeline.
   
   **Analysis Stages:**
   
   1. Input validation
   2. Event detection
   3. Border removal
   4. BIC model selection
   5. Snapshot extraction
   6. Artifact removal
   7. Statistics computation
   8. Causality analysis
   9. Bootstrap analysis
   10. Output preparation
   
   **Key Methods:**
   
   - ``run(original_signal, detection_signal)``: Execute complete pipeline
   - ``analyze(data, **kwargs)``: BaseAnalyzer interface compatibility
   
   **Example:**
   
   .. code-block:: python
   
      from trancit import PipelineOrchestrator
      from trancit.config import PipelineConfig, PipelineOptions, DetectionParams, CausalParams
      
      # Configure pipeline
      config = PipelineConfig(
          options=PipelineOptions(
              detection=True,
              causal_analysis=True,
              bootstrap=False
          ),
          detection=DetectionParams(
              thres_ratio=2.0,
              l_extract=100,
              l_start=50
          ),
          causal=CausalParams(
              ref_time=50,
              estim_mode="OLS"
          )
      )
      
      # Run pipeline
      orchestrator = PipelineOrchestrator(config)
      result = orchestrator.run(original_signal, detection_signal)
      
      # Access results
      if result.results.get("CausalOutput"):
          dcs_results = result.results["CausalOutput"]["OLS"]["DCS"]
          print(f"Detected {len(dcs_results)} events")

.. autoclass:: PipelineResult
   :members:
   :show-inheritance:
   
   Result object containing complete pipeline outputs.
   
   **Attributes:**
   
   - ``results`` (dict): Complete analysis results
   - ``config`` (PipelineConfig): Configuration used for analysis
   - ``event_snapshots`` (ndarray): Extracted event windows

Pipeline Stages
===============

.. currentmodule:: trancit.pipeline.stages

Individual pipeline stages can be used independently for custom workflows.

.. autoclass:: InputValidationStage
   :members:
   :show-inheritance:
   
   Validates input signals and ensures proper format.

.. autoclass:: EventDetectionStage
   :members:
   :show-inheritance:
   
   Detects events in the detection signal based on threshold criteria.

.. autoclass:: BICSelectionStage
   :members:
   :show-inheritance:
   
   Performs BIC-based model order selection.

.. autoclass:: SnapshotExtractionStage
   :members:
   :show-inheritance:
   
   Extracts fixed-length windows around detected events.

.. autoclass:: CausalityAnalysisStage
   :members:
   :show-inheritance:
   
   Computes causality measures for extracted event windows.

**Example of Custom Pipeline:**

.. code-block:: python

   from trancit.pipeline.stages import InputValidationStage, EventDetectionStage, CausalityAnalysisStage
   
   # Create individual stages
   validation_stage = InputValidationStage(config)
   detection_stage = EventDetectionStage(config)
   causality_stage = CausalityAnalysisStage(config)
   
   # Run stages sequentially
   stage_data = {'original_signal': original_signal, 'detection_signal': detection_signal}
   
   validated_data = validation_stage.execute(**stage_data)
   detected_data = detection_stage.execute(**validated_data)
   # ... continue with other stages

*************************************
Configuration (`trancit.config`)
*************************************

The configuration module defines all parameter classes used throughout DCS.

.. currentmodule:: trancit.config

Main Configuration
==================

.. autoclass:: PipelineConfig
   :members:
   :show-inheritance:
   
   Main configuration class that contains all pipeline parameters.
   
   **Components:**
   
   - ``options``: PipelineOptions - controls which analysis stages to run
   - ``detection``: DetectionParams - event detection parameters
   - ``bic``: BicParams - BIC model selection parameters
   - ``causal``: CausalParams - causality analysis parameters
   - ``monte_carlo``: MonteCParams - bootstrap analysis parameters
   - ``output``: OutputParams - output formatting parameters
   
   **Example:**
   
   .. code-block:: python
   
      from trancit.config import *
      
      config = PipelineConfig(
          options=PipelineOptions(
              detection=True,
              bic=True,
              causal_analysis=True,
              bootstrap=False
          ),
          detection=DetectionParams(
              thres_ratio=2.5,
              l_extract=150,
              align_type="peak"
          ),
          bic=BicParams(
              morder=4,
              momax=8,
              mode="OLS"
          ),
          causal=CausalParams(
              ref_time=75,
              estim_mode="OLS",
              diag_flag=False
          )
      )

Parameter Classes
=================

.. autoclass:: PipelineOptions
   :members:
   :show-inheritance:
   
   Controls which analysis stages are enabled.
   
   **Key Parameters:**
   
   - ``detection`` (bool): Enable event detection
   - ``bic`` (bool): Enable BIC model selection
   - ``causal_analysis`` (bool): Enable causality analysis
   - ``bootstrap`` (bool): Enable bootstrap significance testing
   - ``save_flag`` (bool): Save intermediate results
   - ``debiased_stats`` (bool): Use debiased statistics

.. autoclass:: DetectionParams
   :members:
   :show-inheritance:
   
   Parameters for event detection.
   
   **Key Parameters:**
   
   - ``thres_ratio`` (float): Detection threshold (1.5-3.0 typical range)
   - ``l_extract`` (int): Length of extracted event windows
   - ``l_start`` (int): Start offset within windows
   - ``align_type`` (str): "peak" or "onset" alignment
   - ``shrink_flag`` (bool): Apply shrinkage procedure
   - ``remove_artif`` (bool): Remove artifact-contaminated trials

.. autoclass:: CausalParams
   :members:
   :show-inheritance:
   
   Parameters for causality analysis.
   
   **Key Parameters:**
   
   - ``ref_time`` (int): Reference time for rDCS calculation
   - ``estim_mode`` (str): "OLS" or "RLS" estimation
   - ``diag_flag`` (bool): Use diagonal covariance approximation
   - ``old_version`` (bool): Use legacy rDCS calculation method

.. autoclass:: BicParams
   :members:
   :show-inheritance:
   
   Parameters for BIC model selection.

.. autoclass:: MonteCParams
   :members:
   :show-inheritance:
   
   Parameters for bootstrap analysis.

.. autoclass:: OutputParams
   :members:
   :show-inheritance:
   
   Parameters for output formatting.

***********************************
Model Estimation (`trancit.models`)
***********************************

The models module provides VAR model estimation and selection tools.

VAR Model Estimation
====================

.. currentmodule:: trancit.models.var_estimation

.. autoclass:: VAREstimator
   :members:
   :show-inheritance:
   
   Vector Autoregressive (VAR) model estimation using OLS or RLS methods.
   
   **Example:**
   
   .. code-block:: python
   
      from trancit.models import VAREstimator
      
      estimator = VAREstimator(model_order=4, method="OLS")
      coefficients, residuals = estimator.fit(data)

BIC Model Selection
===================

.. currentmodule:: trancit.models.bic_selection

.. autoclass:: BICSelector
   :members:
   :show-inheritance:
   
   Bayesian Information Criterion (BIC) for automatic model order selection.
   
   **Example:**
   
   .. code-block:: python
   
      from trancit.models import BICSelector
      
      selector = BICSelector(max_order=10, mode="OLS")
      bic_results = selector.compute_multi_trial_BIC(data, params)
      optimal_order = bic_results['morder']

Model Validation
================

.. currentmodule:: trancit.models.model_validation

.. autoclass:: ModelValidator
   :members:
   :show-inheritance:
   
   Tools for validating fitted VAR models.

******************************
Simulation (`trancit.simulation`)
******************************

The simulation module provides tools for generating synthetic time series data for testing and validation.

Signal Generation
=================

.. currentmodule:: trancit.simulation

.. autofunction:: generate_signals

   Generate synthetic bivariate coupled oscillator signals.
   
   **Parameters:**
   
   - ``T`` (int): Number of time points
   - ``Ntrial`` (int): Number of trials
   - ``h`` (float): Time step
   - ``gamma1``, ``gamma2`` (float): Damping parameters
   - ``Omega1``, ``Omega2`` (float): Natural frequencies
   
   **Returns:**
   
   Tuple of (data, time_axis, parameters) where data has shape (2, T, Ntrial).
   
   **Example:**
   
   .. code-block:: python
   
      from trancit import generate_signals
      
      data, time_axis, params = generate_signals(
          T=1000, Ntrial=20, h=0.1,
          gamma1=0.5, gamma2=0.5,
          Omega1=1.0, Omega2=1.2
      )
      
      print(f"Generated data shape: {data.shape}")

AR Event Simulation
===================

.. autofunction:: simulate_ar_event

   Generate autoregressive events with controlled causality.

.. autofunction:: simulate_ar_event_bootstrap

   Generate multiple bootstrap samples of AR events.

VAR Simulation
==============

.. autofunction:: generate_var_nonstat

   Generate non-stationary VAR processes.

**************************
Utilities (`trancit.utils`)
**************************

The utilities module provides helper functions for data processing, analysis, and visualization.

Data Processing
===============

.. currentmodule:: trancit.utils.preprocess

.. autofunction:: normalize_data

   Normalize time series data using various methods.
   
   **Parameters:**
   
   - ``data`` (ndarray): Input data
   - ``method`` (str): "zscore", "minmax", or "robust"
   - ``axis`` (int): Axis along which to normalize
   
   **Example:**
   
   .. code-block:: python
   
      from trancit.utils.preprocess import normalize_data
      
      normalized_data = normalize_data(data, method="zscore", axis=1)

.. autofunction:: check_data_quality

   Check data quality and identify potential issues.

Core Analysis Functions
=======================

.. currentmodule:: trancit.utils.core

.. autofunction:: compute_event_statistics

   Compute statistics for event snapshots.

.. autofunction:: perform_desnap_analysis

   Perform debiased snapshot analysis.

Signal Processing
=================

.. currentmodule:: trancit.utils.signal

.. autofunction:: find_peak_locations

   Find peak locations in signals for event detection.

.. autofunction:: extract_event_snapshots

   Extract event snapshots from continuous signals.

Residual Analysis
=================

.. currentmodule:: trancit.utils.residuals

.. autofunction:: estimate_residuals

   Estimate VAR model residuals.

.. autofunction:: get_residuals

   Get residuals from fitted models.

Visualization
=============

.. currentmodule:: trancit.utils.plotting

.. autofunction:: plot_causality_results

   Plot causality analysis results.
   
   **Example:**
   
   .. code-block:: python
   
      from trancit.utils.plotting import plot_causality_results
      
      # Assuming 'result' is a DCSResult object
      plot_causality_results(
          result.causal_strength,
          result.transfer_entropy,
          result.granger_causality,
          title="DCS Analysis Results"
      )

Helper Functions
================

.. currentmodule:: trancit.utils.helpers

.. autofunction:: remove_artifact_trials

   Remove trials contaminated by artifacts.

.. autofunction:: shrink_locations_resample_uniform

   Apply shrinkage procedure to event locations.

.. autofunction:: find_best_shrinked_locations

   Find optimal shrinkage parameters.

*************************
Complete Usage Examples
*************************

Basic Analysis Workflow
========================

.. code-block:: python

   # Complete workflow example
   import numpy as np
   from trancit import DCSCalculator, generate_signals
   from trancit.utils.preprocess import normalize_data
   import matplotlib.pyplot as plt
   
   # 1. Generate or load data
   data, _, _ = generate_signals(T=500, Ntrial=20, h=0.1,
                                gamma1=0.5, gamma2=0.5,
                                Omega1=1.0, Omega2=1.2)
   
   # 2. Preprocess data
   clean_data = normalize_data(data, method="zscore", axis=1)
   
   # 3. Perform analysis
   calculator = DCSCalculator(model_order=4, time_mode="inhomo")
   result = calculator.analyze(clean_data)
   
   # 4. Examine results
   print(f"Mean X→Y causality: {result.causal_strength[:, 1].mean():.4f}")
   print(f"Mean Y→X causality: {result.causal_strength[:, 0].mean():.4f}")
   
   # 5. Visualize results
   plt.figure(figsize=(12, 4))
   plt.subplot(1, 2, 1)
   plt.plot(result.causal_strength[:, 1], label='X→Y', linewidth=2)
   plt.plot(result.causal_strength[:, 0], label='Y→X', linewidth=2)
   plt.xlabel('Time')
   plt.ylabel('Causal Strength')
   plt.legend()
   plt.title('Dynamic Causal Strength')
   
   plt.subplot(1, 2, 2)
   plt.plot(result.transfer_entropy[:, 1], label='X→Y', linewidth=2)
   plt.plot(result.transfer_entropy[:, 0], label='Y→X', linewidth=2)
   plt.xlabel('Time')
   plt.ylabel('Transfer Entropy')
   plt.legend()
   plt.title('Transfer Entropy')
   
   plt.tight_layout()
   plt.show()

Event-Based Analysis Workflow
==============================

.. code-block:: python

   # Complete event-based analysis
   from trancit import PipelineOrchestrator, generate_signals
   from trancit.config import *
   
   # Generate data
   data, _, _ = generate_signals(T=800, Ntrial=15, h=0.1,
                                gamma1=0.4, gamma2=0.6,
                                Omega1=0.9, Omega2=1.3)
   
   original_signal = np.mean(data, axis=2)
   detection_signal = original_signal * 1.5
   
   # Configure pipeline
   config = PipelineConfig(
       options=PipelineOptions(
           detection=True,
           bic=False,
           causal_analysis=True,
           bootstrap=False
       ),
       detection=DetectionParams(
           thres_ratio=2.0,
           l_extract=100,
           l_start=50,
           align_type="peak"
       ),
       causal=CausalParams(
           ref_time=50,
           estim_mode="OLS"
       )
   )
   
   # Run analysis
   orchestrator = PipelineOrchestrator(config)
   result = orchestrator.run(original_signal, detection_signal)
   
   # Process results
   if result.results.get("CausalOutput"):
       causal_output = result.results["CausalOutput"]["OLS"]
       
       if "DCS" in causal_output:
           dcs_results = causal_output["DCS"]
           print(f"Events detected: {len(dcs_results)}")
           print(f"Mean event-wise X→Y causality: {dcs_results[:, 1].mean():.4f}")
           
           # Plot per-event causality
           plt.figure(figsize=(10, 6))
           event_numbers = range(1, len(dcs_results) + 1)
           
           plt.bar(event_numbers, dcs_results[:, 1], alpha=0.7, label='X→Y')
           plt.bar(event_numbers, dcs_results[:, 0], alpha=0.7, label='Y→X')
           plt.xlabel('Event Number')
           plt.ylabel('Causal Strength')
           plt.legend()
           plt.title('Event-Based Causality Analysis')
           plt.show()

*******************
Performance Notes
*******************

**Memory Usage:**

- DCS analysis memory scales approximately as O(T × N × M²) where T=time points, N=trials, M=model order
- For large datasets, consider processing in chunks or using ``use_diagonal_covariance=True``

**Computation Time:**

- Typical analysis time: 0.1-10 seconds depending on data size
- Time-inhomogeneous mode is slower but more accurate for non-stationary data
- BIC model selection adds significant computation time but improves model quality

**Optimization Tips:**

1. Use ``time_mode="homo"`` for stationary data (faster)
2. Enable ``use_diagonal_covariance=True`` for numerical stability with highly correlated data
3. Choose model order carefully: higher order = more accurate but slower
4. Normalize data to improve numerical stability
5. Use multiple trials to improve statistical robustness

*******************
Troubleshooting
*******************

**Common Issues:**

1. **"Input data must be bivariate"**
   - Ensure data shape is (2, time_points, trials)
   
2. **Singular matrix errors**
   - Try ``use_diagonal_covariance=True``
   - Check data for perfect correlations
   - Normalize data using ``normalize_data()``
   
3. **"Insufficient observations"**
   - Reduce model order
   - Collect more data
   - Use ``time_mode="homo"`` for short series
   
4. **Pipeline finds no events**
   - Lower ``thres_ratio`` parameter
   - Check detection signal characteristics
   - Visualize signals to understand event structure

**Getting Help:**

- Check the tutorials in :doc:`tutorials`
- Review examples in the ``examples/`` directory
- Open an issue on `GitHub <https://github.com/CMC-lab/TranCIT/issues>`_
- Read the scientific paper for theoretical background
