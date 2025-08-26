"""
Core module for TranCIT: Transient Causal Interaction.

This module provides the foundation classes and interfaces
that are used throughout the TranCIT package.

The core module includes:
- BaseAnalyzer: Abstract base class for all analyzers
- BaseResult: Base class for all result objects
- BaseConfig: Base class for configuration objects
- Custom exceptions: DCSError, ValidationError, ComputationError, etc.

Example
-------
>>> from trancit.core import BaseAnalyzer, BaseResult
>>>
>>> class MyAnalyzer(BaseAnalyzer):
...     def analyze(self, data):
...         return BaseResult(result=data.sum())
>>>
>>> analyzer = MyAnalyzer(param1=1, param2=2)
>>> result = analyzer.analyze(np.array([1, 2, 3]))
"""

from .base import BaseAnalyzer, BaseResult
from .exceptions import ComputationError, DCSError, ValidationError

__all__ = [
    "BaseAnalyzer",
    "BaseResult",
    "DCSError",
    "ValidationError",
    "ComputationError",
]
