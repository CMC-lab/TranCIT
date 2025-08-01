"""
Base classes for Dynamic Causal Strength (DCS).

This module provides the foundation classes and interfaces
that define the common structure for all DCS components.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional

import numpy as np

from .exceptions import ValidationError


class BaseResult:
    """Base class for all result objects."""
    
    def __init__(self, **kwargs):
        """Initialize result with keyword arguments."""
        for key, value in kwargs.items():
            setattr(self, key, value)
    
    def __repr__(self) -> str:
        """String representation of the result."""
        attrs = [f"{k}={v}" for k, v in self.__dict__.items()]
        return f"{self.__class__.__name__}({', '.join(attrs)})"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert result to dictionary."""
        return self.__dict__.copy()


class BaseAnalyzer(ABC):
    """
    Abstract base class for all analyzers.
    
    This class defines the common interface and functionality
    for all analysis components in the DCS package.
    """
    
    def __init__(self, **kwargs):
        """Initialize analyzer with configuration."""
        self.config = kwargs
        self._validate_config()
    
    @abstractmethod
    def analyze(self, data: np.ndarray, **kwargs) -> BaseResult:
        """
        Perform analysis on the given data.
        
        Parameters
        ----------
        data : np.ndarray
            Input data for analysis
        **kwargs
            Additional parameters specific to the analyzer
            
        Returns
        -------
        BaseResult
            Analysis results
        """
        pass
    
    def _validate_config(self) -> None:
        """Validate configuration parameters."""
        # Override in subclasses for specific validation
        pass
    
    def _validate_input_data(self, data: np.ndarray) -> None:
        """Validate input data format and dimensions."""
        if not isinstance(data, np.ndarray):
            raise ValidationError("Input data must be a NumPy array")
        
        if data.size == 0:
            raise ValidationError("Input data cannot be empty")
        
        if np.any(np.isnan(data)) or np.any(np.isinf(data)):
            raise ValidationError("Input data contains NaN or Inf values")
    
    def _log_analysis_start(self, data_shape: tuple) -> None:
        """Log the start of analysis."""
        print(f"Starting {self.__class__.__name__} analysis on data with shape {data_shape}")
    
    def _log_analysis_complete(self) -> None:
        """Log the completion of analysis."""
        print(f"Completed {self.__class__.__name__} analysis")


class BaseConfig:
    """Base class for configuration objects."""
    
    def __init__(self, **kwargs):
        """Initialize configuration with keyword arguments."""
        for key, value in kwargs.items():
            setattr(self, key, value)
    
    def validate(self) -> None:
        """Validate configuration parameters."""
        # Override in subclasses for specific validation
        pass
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        return self.__dict__.copy()
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'BaseConfig':
        """Create configuration from dictionary."""
        return cls(**config_dict) 
