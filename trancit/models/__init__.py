"""
Models module for TranCIT: Transient Causal Interaction.

This module provides implementations for:
- VAR model estimation: Vector Autoregressive model fitting
- BIC model selection: Bayesian Information Criterion for model order selection
- Model validation: Diagnostics and validation of fitted models
- Model selection: High-level model selection utilities

All model estimators provide consistent interfaces for time series modeling
and support both homogeneous and inhomogeneous VAR models.

Example
-------
>>> import numpy as np
>>> from trancit.models import VAREstimator, BICSelector, select_model_order
>>>
>>> # Create VAR estimator
>>> estimator = VAREstimator(model_order=4, time_mode="inhomo")
>>>
>>> # Generate sample data
>>> data = np.random.randn(2, 1000, 20)  # (n_vars, n_obs, n_trials)
>>>
>>> # Estimate VAR coefficients
>>> coefficients, residuals, log_likelihood, hessian_sum = estimator.estimate_var_coefficients(
...     data, max_model_order=4
... )
>>>
>>> # Select optimal model order
>>> bic_scores, optimal_orders, log_likelihoods, penalty_terms = select_model_order(
...     data, max_model_order=6, time_mode="inhomo"
... )
>>>
>>> print(f"Coefficients shape: {coefficients.shape}")
>>> print(f"Optimal model orders: {optimal_orders}")
"""

from .bic_selection import BICSelector
from .model_selection import select_model_order
from .model_validation import ModelValidator
from .var_estimation import VAREstimator

__all__ = [
    "VAREstimator",
    "BICSelector",
    "ModelValidator",
    "select_model_order",
]
