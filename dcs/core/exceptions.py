"""
Custom exceptions for Dynamic Causal Strength (DCS).

This module defines the exception hierarchy used throughout
the DCS package for consistent error handling.
"""


class DCSError(Exception):
    """Base exception for all DCS-related errors."""

    def __init__(self, message: str, details: str = None):
        """
        Initialize DCS error.

        Parameters
        ----------
        message : str
            Error message
        details : str, optional
            Additional error details
        """
        self.message = message
        self.details = details
        super().__init__(self.message)


class ValidationError(DCSError):
    """Exception raised when input validation fails."""

    def __init__(self, message: str, field: str = None, value: any = None):
        """
        Initialize validation error.

        Parameters
        ----------
        message : str
            Validation error message
        field : str, optional
            Name of the field that failed validation
        value : any, optional
            Value that failed validation
        """
        self.field = field
        self.value = value
        super().__init__(message, f"Field: {field}, Value: {value}")


class ComputationError(DCSError):
    """Exception raised when computation fails."""

    def __init__(self, message: str, step: str = None, data_shape: tuple = None):
        """
        Initialize computation error.

        Parameters
        ----------
        message : str
            Computation error message
        step : str, optional
            Step in the computation that failed
        data_shape : tuple, optional
            Shape of the data being processed
        """
        self.step = step
        self.data_shape = data_shape
        super().__init__(message, f"Step: {step}, Data shape: {data_shape}")


class ConfigurationError(DCSError):
    """Exception raised when configuration is invalid."""

    def __init__(self, message: str, config_key: str = None, expected_type: str = None):
        """
        Initialize configuration error.

        Parameters
        ----------
        message : str
            Configuration error message
        config_key : str, optional
            Configuration key that caused the error
        expected_type : str, optional
            Expected type for the configuration value
        """
        self.config_key = config_key
        self.expected_type = expected_type
        super().__init__(message, f"Key: {config_key}, Expected type: {expected_type}")


class DataError(DCSError):
    """Exception raised when data is invalid or corrupted."""

    def __init__(self, message: str, data_shape: tuple = None, data_type: str = None):
        """
        Initialize data error.

        Parameters
        ----------
        message : str
            Data error message
        data_shape : tuple, optional
            Shape of the problematic data
        data_type : str, optional
            Type of the problematic data
        """
        self.data_shape = data_shape
        self.data_type = data_type
        super().__init__(message, f"Shape: {data_shape}, Type: {data_type}")


class ConvergenceError(ComputationError):
    """Exception raised when numerical algorithms fail to converge."""

    def __init__(
        self, message: str, max_iterations: int = None, tolerance: float = None
    ):
        """
        Initialize convergence error.

        Parameters
        ----------
        message : str
            Convergence error message
        max_iterations : int, optional
            Maximum iterations attempted
        tolerance : float, optional
            Tolerance used for convergence check
        """
        self.max_iterations = max_iterations
        self.tolerance = tolerance
        super().__init__(message, "convergence", None)


class SingularMatrixError(ComputationError):
    """Exception raised when encountering singular matrices."""

    def __init__(
        self, message: str, matrix_shape: tuple = None, condition_number: float = None
    ):
        """
        Initialize singular matrix error.

        Parameters
        ----------
        message : str
            Singular matrix error message
        matrix_shape : tuple, optional
            Shape of the singular matrix
        condition_number : float, optional
            Condition number of the matrix
        """
        self.matrix_shape = matrix_shape
        self.condition_number = condition_number
        super().__init__(message, "matrix_inversion", matrix_shape)
