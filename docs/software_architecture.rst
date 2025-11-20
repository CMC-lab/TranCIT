.. _software_architecture:

##########################
Software Architecture
##########################

This document provides detailed information about the TranCIT package's software architecture, dependencies, and design choices.

Package Structure
=================

TranCIT follows a modular, object-oriented design with clear separation of concerns. The package is organized into six main modules:

Core Module (`trancit.core`)
----------------------------

The core module provides the foundational classes and interfaces used throughout the package:

- **`BaseAnalyzer`**: Abstract base class defining the common interface for all analysis components. All calculators and analyzers inherit from this class, ensuring consistent API design across the package.

- **`BaseResult`**: Base class for all result objects, providing a standardized structure for returning analysis results.

- **`BaseConfig`**: Base configuration class that enables consistent parameter management and validation.

- **Exception Hierarchy**: Comprehensive exception classes (`ValidationError`, `ComputationError`, `ConvergenceError`, `SingularMatrixError`, etc.) that provide clear error messages and enable graceful error recovery.

**Design Rationale:** The base class pattern ensures that all analysis components share a common interface, making the API predictable and extensible. Users can create custom analyzers by inheriting from `BaseAnalyzer` without modifying core code.

Causality Module (`trancit.causality`)
---------------------------------------

Implements the four primary causality methods as independent calculator classes:

- **`DCSCalculator`**: Computes Dynamic Causal Strength, a time-varying measure of direct causal influence based on structural causal models.

- **`RelativeDCSCalculator`**: Computes relative Dynamic Causal Strength, comparing causal effects to a baseline reference period.

- **`TransferEntropyCalculator`**: Computes Transfer Entropy, an information-theoretic measure quantifying directed information flow.

- **`GrangerCausalityCalculator`**: Computes Granger Causality, a linear measure testing whether past values of one series help predict another.

Each calculator inherits from `BaseAnalyzer` and returns standardized result objects, ensuring consistent usage patterns across all methods.

**Design Rationale:** Independent calculator classes allow users to use specific causality methods without loading unnecessary code, improving performance and maintainability.

Pipeline Module (`trancit.pipeline`)
-------------------------------------

Implements a stage-based pipeline architecture using the Pipeline Pattern for event-based analysis workflows. The module provides:

- **`PipelineOrchestrator`**: Main coordinator that executes preprocessing stages sequentially, manages pipeline state, and handles error propagation.

- **`PipelineStage`**: Abstract base class defining the interface for all preprocessing stages.

- **Individual Stage Classes**: Modular stages for event detection, border removal, BIC selection, snapshot extraction, artifact removal, statistics computation, causality analysis, bootstrap analysis, and output preparation.

**Design Rationale:** The pipeline architecture enables flexible, composable workflows. Users can customize preprocessing by configuring individual stages or execute stages independently for fine-grained control.

**Note:** Detailed documentation of the pipeline architecture, including state management and API design, is available in :doc:`event_detection_preprocessing`.

Models Module (`trancit.models`)
----------------------------------

Provides VAR model estimation and validation:

- **`VAREstimator`**: Estimates Vector Autoregressive model coefficients using Ordinary Least Squares (OLS) or other estimation methods.

- **`BICSelector`**: Performs Bayesian Information Criterion-based model order selection to automatically determine optimal VAR model complexity.

- **`ModelValidator`**: Validates VAR model assumptions and checks for numerical stability.

**Design Rationale:** Separating model estimation from causality computation allows reuse of VAR models across different causality methods and enables independent testing and optimization.

Simulation Module (`trancit.simulation`)
-----------------------------------------

Contains functions for generating synthetic time series data:

- **`generate_signals`**: Generates coupled oscillator time series with known causal structures.

- **`simulate_ar_event`**: Generates autoregressive events with controlled causality.

- **`simulate_ar_event_bootstrap`**: Generates multiple bootstrap samples of AR events.

**Design Rationale:** Synthetic data generation is essential for validation, testing, and educational purposes. Separating simulation into its own module keeps the codebase organized and allows independent development.

Utilities Module (`trancit.utils`)
-----------------------------------

Provides helper functions for data preprocessing, signal processing, visualization, and statistical computations:

- **`core`**: Core utility functions for event window extraction, statistics computation, and data manipulation.

- **`preprocess`**: Preprocessing utilities for artifact removal and data cleaning.

- **`signal`**: Signal processing functions for peak detection, location shrinking, and signal alignment.

- **`plotting`**: Visualization utilities for plotting causality results and time series data.

- **`residuals`**: Functions for computing and analyzing VAR model residuals.

**Design Rationale:** Utility functions are organized by functionality to improve code discoverability and maintainability. This modular structure allows users to import only the utilities they need.

Dependencies
============

Core Dependencies
-----------------

**NumPy ≥1.19.5**
   - **Purpose:** Fundamental array operations and numerical computations
   - **Usage:** All data structures use NumPy arrays. Core computations rely on NumPy's vectorized operations for performance.

**SciPy ≥1.7.0**
   - **Purpose:** Statistical functions, optimization, and signal processing
   - **Usage:** Used for statistical computations, filtering operations, and numerical optimization in VAR model estimation.

**scikit-learn ≥1.0.0**
   - **Purpose:** Covariance estimation (specifically Ledoit-Wolf shrinkage)
   - **Usage:** Provides robust covariance matrix estimation for VAR model residuals, improving numerical stability in high-dimensional settings.

**matplotlib ≥3.5.0**
   - **Purpose:** Visualization (optional dependency for plotting utilities)
   - **Usage:** Used in plotting utilities for visualizing causality results, time series data, and analysis outputs. The package can function without matplotlib if visualization is not needed.

Design Choices
==============

Configuration-Driven Architecture
----------------------------------

All analysis parameters are specified through dataclass-based configuration objects (`PipelineConfig`, `DetectionParams`, `CausalParams`, `BicParams`, etc.). This design choice provides:

- **Type Safety:** Dataclasses enable static type checking and IDE autocompletion
- **Clear Documentation:** Each parameter is documented in the dataclass definition
- **Easy Validation:** Parameters can be validated at configuration time
- **Reproducibility:** Configuration objects can be serialized and saved for exact result replication

**Note:** Detailed examples of configuration usage, including preprocessing pipeline configuration, are available in :doc:`event_detection_preprocessing`.

Extensibility Through Abstract Base Classes
--------------------------------------------

The package uses abstract base classes (`BaseAnalyzer`, `PipelineStage`) to define interfaces that users can implement for custom functionality:

- **Custom Analyzers:** Users can create new causality methods by inheriting from `BaseAnalyzer`
- **Custom Pipeline Stages:** Users can add new preprocessing stages by implementing `PipelineStage`
- **No Core Modification Required:** Extensions work alongside existing code without modifying package internals

**Example:**
.. code-block:: python

   from trancit.core.base import BaseAnalyzer, BaseResult

   class MyCustomAnalyzer(BaseAnalyzer):
       def analyze(self, data, **kwargs):
           # Custom analysis logic
           return MyCustomResult(...)

Comprehensive Error Handling
------------------------------

The package defines a hierarchy of custom exceptions that provide context-specific error messages:

- **`ValidationError`**: Raised when input data or parameters are invalid
- **`ComputationError`**: Raised when numerical computations fail
- **`ConvergenceError`**: Raised when iterative algorithms fail to converge
- **`SingularMatrixError`**: Raised when matrix operations encounter singular matrices
- **`DataError`**: Raised when data format or structure is incorrect
- **`ConfigurationError`**: Raised when configuration parameters are inconsistent

This design enables:
- **Clear Error Messages:** Users receive specific information about what went wrong
- **Graceful Degradation:** Errors can be caught and handled appropriately
- **Debugging Support:** Error types help identify the source of problems

Reproducibility and Logging
----------------------------

All analysis steps are logged using Python's `logging` module, with configurable verbosity levels:

- **INFO Level:** Default logging provides detailed progress information
- **WARNING Level:** Shows only warnings and errors
- **DEBUG Level:** Provides extensive debugging information

The pipeline state is traceable through sequential stage execution, allowing users to debug issues, verify stage execution, and reproduce analyses by following logged steps.

Testing and Quality Assurance
==============================

The package includes comprehensive testing infrastructure:

- **Test Coverage:** Using `pytest`
- **Unit Tests:** Individual components are tested in isolation
- **Integration Tests:** Full pipeline workflows are tested end-to-end
- **Continuous Integration:** GitHub Actions runs tests across Python versions (3.9-3.13) and platforms
- **Type Hints:** Full type annotation throughout the codebase for improved maintainability and IDE support

Performance Considerations
==========================

- **Vectorized Operations:** Uses NumPy vectorization for efficient array operations
- **Memory Efficiency:** Processes data in-place where possible
- **Lazy Evaluation:** Optional stages (BIC, bootstrap) are only executed when enabled
- **Parallelization Ready:** Stage-based design allows for future parallel execution of independent stages

Version Management
==================

Version numbers are automatically managed using `setuptools_scm` from git tags, ensuring version consistency across the package and documentation.

