
.. _api_reference:

############
API Reference
############

Welcome to the API reference for **Dynamic Causal Strength (DCS)**. This section provides detailed, auto-generated documentation for all modules in the package.

The DCS API is organized into several core components:

- The main `dcs` package
- Functional submodules like `pipeline`, `simulation`, `causality`, and `models`
- Utility modules under `dcs.utils` for signal processing, preprocessing, and more

All public classes and functions are documented here. DCS uses **Google-style** or **NumPy-style** docstrings for clarity and compatibility with `sphinx.ext.napoleon`.

Main Package
============
This is the main entry point to core functionality and high-level access.

.. automodule:: dcs
   :members:
   :show-inheritance:

Pipeline Module
===============
Handles the full causal detection pipeline including configuration and orchestration.

.. automodule:: dcs.pipeline
   :members:
   :show-inheritance:

Causality Module
================
Provides causal analysis logic (e.g., DCS estimators).

.. automodule:: dcs.causality
   :members:
   :show-inheritance:

Models Module
=============
Time-series model definitions and parameter interfaces.

.. automodule:: dcs.models
   :members:
   :show-inheritance:

Simulation Module
=================
Synthetic data generation utilities for testing and evaluation.

.. automodule:: dcs.simulation
   :members:
   :show-inheritance:

Utilities (`dcs.utils`)
=======================

Core Utilities
--------------
Low-level operations like matrix computation and stability checks.

.. automodule:: dcs.utils.core
   :members:
   :show-inheritance:

Helper Utilities
----------------
General-purpose functions used across the package.

.. automodule:: dcs.utils.helpers
   :members:
   :show-inheritance:

Preprocessing Utilities
-----------------------
Signal conditioning and preparation for event detection.

.. automodule:: dcs.utils.preprocess
   :members:
   :show-inheritance:

Residual Utilities
------------------
Extracting and analyzing residuals for model evaluation.

.. automodule:: dcs.utils.residuals
   :members:
   :show-inheritance:

Signal Utilities
----------------
Signal-specific helpers for time-series alignment, filtering, and more.

.. automodule:: dcs.utils.signal
   :members:
   :show-inheritance:
