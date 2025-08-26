"""
Preprocessing utilities for Dynamic Causal Strength (DCS).

This module provides functions for data cleaning, artifact removal, and matrix regularization.
These utilities ensure data quality and numerical stability for the DCS pipeline.
"""

import logging
from typing import Optional, Tuple

import numpy as np
from sklearn.covariance import ledoit_wolf

from ..core.exceptions import ComputationError, ValidationError

logger = logging.getLogger(__name__)


def remove_artifact_trials(
    event_data: np.ndarray, locations: np.ndarray, lower_threshold: float
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Remove trials from event data where the signal drops below a specified threshold.

    This function identifies and removes trials where any value in the first two variables
    of the event data falls below the given lower threshold. It also removes the corresponding
    locations from the `locations` array.

    Parameters
    ----------
    event_data : np.ndarray
        3D array of shape (variables, time_points, trials) containing the event data.
    locations : np.ndarray
        1D array of shape (trials,) containing location indices for each trial.
    lower_threshold : float
        The threshold value below which trials are considered artifacts and removed.

    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        A tuple containing:
        - The updated event_data with artifact trials removed.
        - The updated locations array with corresponding entries removed.

    Raises
    ------
    ValidationError
        If input parameters are invalid.
    ComputationError
        If artifact removal fails.

    Examples
    --------
    >>> import numpy as np
    >>> event_data = np.random.randn(2, 100, 10)  # (vars, time, trials)
    >>> locations = np.array([100, 200, 300, 400, 500, 600, 700, 800, 900, 1000])
    >>> cleaned_data, cleaned_locs = remove_artifact_trials(event_data, locations, -15000)
    >>> print(f"Original trials: {event_data.shape[2]}")
    >>> print(f"Cleaned trials: {cleaned_data.shape[2]}")
    """
    # Input validation
    if not isinstance(event_data, np.ndarray) or event_data.ndim != 3:
        raise ValidationError(
            "event_data must be a 3D numpy array",
            "event_data_ndim",
            event_data.ndim if hasattr(event_data, "ndim") else None,
        )

    if not isinstance(locations, np.ndarray) or locations.ndim != 1:
        raise ValidationError(
            "locations must be a 1D numpy array",
            "locations_ndim",
            locations.ndim if hasattr(locations, "ndim") else None,
        )

    if not isinstance(lower_threshold, (int, float)):
        raise ValidationError(
            "lower_threshold must be a number",
            "lower_threshold_type",
            type(lower_threshold),
        )

    # Check dimensions
    if event_data.shape[2] != len(locations):
        raise ValidationError(
            "event_data trials dimension must match locations length",
            "dimension_mismatch",
            (event_data.shape[2], len(locations)),
        )

    if event_data.shape[0] < 2:
        raise ValidationError(
            "event_data must have at least 2 variables", "n_vars", event_data.shape[0]
        )

    try:
        artifact_mask = np.any(event_data[:2, :, :] < lower_threshold, axis=(0, 1))
        trials_to_remove = np.where(artifact_mask)[0]

        if len(trials_to_remove) == 0:
            logger.info("No artifact trials found")
            return event_data, locations

        updated_event_data = np.delete(event_data, trials_to_remove, axis=2)
        updated_locations = np.delete(locations, trials_to_remove)

        logger.info(
            f"Removed {len(trials_to_remove)} artifact trials (threshold: {lower_threshold})"
        )
        logger.info(f"Remaining trials: {updated_event_data.shape[2]}")

        return updated_event_data, updated_locations

    except Exception as e:
        logger.error(f"Artifact removal failed: {e}")
        raise ComputationError(f"Artifact removal failed: {e}", "artifact_removal", ())


def regularize_if_singular(
    matrix: np.ndarray,
    samples: Optional[int] = None,
    epsilon: float = 1e-4,
    threshold: float = 1e-4,
) -> np.ndarray:
    """
    Check if a matrix is singular and regularize it by adding epsilon to the diagonal if needed.

    This function checks if the determinant of the matrix is below a specified threshold.
    If it is, the matrix is considered singular, and a small value (epsilon)
    is added to its diagonal
    to make it invertible. Otherwise, the original matrix is returned.

    Parameters
    ----------
    matrix : np.ndarray
        Square matrix to check and potentially regularize.
    samples : Optional[int], optional
        Number of samples for Ledoit-Wolf regularization (if provided).
    epsilon : float, optional
        Small value to add to the diagonal if the matrix is singular. Default is 1e-6.
    threshold : float, optional
        Determinant threshold below which the matrix is considered singular. Default is 1e-6.

    Returns
    -------
    np.ndarray
        The original matrix if non-singular, or the regularized matrix if singular.

    Raises
    ------
    ValidationError
        If input parameters are invalid.
    ComputationError
        If regularization fails.

    Examples
    --------
    >>> import numpy as np
    >>> matrix = np.array([[1, 0], [0, 0]])  # Singular matrix
    >>> regularized = regularize_if_singular(matrix, epsilon=1e-6)
    >>> print(f"Original determinant: {np.linalg.det(matrix)}")
    >>> print(f"Regularized determinant: {np.linalg.det(regularized)}")
    """
    if not isinstance(matrix, np.ndarray) or matrix.ndim != 2:
        raise ValidationError(
            "matrix must be a 2D numpy array",
            "matrix_ndim",
            matrix.ndim if hasattr(matrix, "ndim") else None,
        )

    if matrix.shape[0] != matrix.shape[1]:
        raise ValidationError("matrix must be square", "matrix_shape", matrix.shape)

    if not isinstance(epsilon, (int, float)) or epsilon <= 0:
        raise ValidationError("epsilon must be a positive number", "epsilon", epsilon)

    if not isinstance(threshold, (int, float)) or threshold <= 0:
        raise ValidationError(
            "threshold must be a positive number", "threshold", threshold
        )

    if samples is not None and (not isinstance(samples, int) or samples <= 0):
        raise ValidationError("samples must be a positive integer", "samples", samples)

    try:
        det = np.linalg.det(matrix)

        if abs(det) < threshold:
            logger.warning(
                f"Singular matrix detected (det={det:.2e}), applying regularization"
            )

            if samples is not None and samples > matrix.shape[0]:
                try:
                    regularized_matrix = ledoit_wolf(matrix, assume_centered=False)[0]
                    logger.info("Applied Ledoit-Wolf regularization")
                    return regularized_matrix
                except Exception as e:
                    logger.warning(
                        f"Ledoit-Wolf regularization failed: {e}, using diagonal regularization"
                    )

            regularized_matrix = matrix + epsilon * np.eye(matrix.shape[0])
            logger.info(f"Applied diagonal regularization with epsilon={epsilon}")
            return regularized_matrix
        else:
            return matrix

    except Exception as e:
        logger.error(f"Matrix regularization failed: {e}")
        raise ComputationError(
            f"Matrix regularization failed: {e}", "matrix_regularization", ()
        )


def validate_data_quality(
    data: np.ndarray,
    check_nan: bool = True,
    check_inf: bool = True,
    check_constant: bool = True,
) -> Tuple[bool, str]:
    """
    Validate data quality by checking for common issues.

    This function performs various quality checks on the input data to ensure
    it's suitable for analysis.

    Parameters
    ----------
    data : np.ndarray
        Input data to validate.
    check_nan : bool, optional
        Whether to check for NaN values. Default is True.
    check_inf : bool, optional
        Whether to check for infinite values. Default is True.
    check_constant : bool, optional
        Whether to check for constant variables. Default is True.

    Returns
    -------
    Tuple[bool, str]
        A tuple containing:
        - is_valid : bool
            True if data passes all quality checks.
        - message : str
            Description of any issues found.

    Examples
    --------
    >>> import numpy as np
    >>> data = np.random.randn(2, 100, 10)
    >>> is_valid, message = validate_data_quality(data)
    >>> print(f"Data valid: {is_valid}")
    >>> print(f"Message: {message}")
    """
    # Input validation
    if not isinstance(data, np.ndarray):
        return False, "Input must be a numpy array"

    if not isinstance(check_nan, bool):
        raise ValidationError(
            "check_nan must be a boolean", "check_nan_type", type(check_nan)
        )

    if not isinstance(check_inf, bool):
        raise ValidationError(
            "check_inf must be a boolean", "check_inf_type", type(check_inf)
        )

    if not isinstance(check_constant, bool):
        raise ValidationError(
            "check_constant must be a boolean",
            "check_constant_type",
            type(check_constant),
        )

    issues = []

    try:
        if check_nan and np.any(np.isnan(data)):
            nan_count = np.sum(np.isnan(data))
            issues.append(f"Found {nan_count} NaN values")

        if check_inf and np.any(np.isinf(data)):
            inf_count = np.sum(np.isinf(data))
            issues.append(f"Found {inf_count} infinite values")

        if check_constant and data.ndim >= 2:
            for i in range(data.shape[0]):
                if np.all(data[i] == data[i].flat[0]):
                    issues.append(f"Variable {i} is constant")

        if data.size == 0:
            issues.append("Data is empty")

        if data.size > 0:
            max_val = np.max(np.abs(data))
            if max_val > 1e10:
                issues.append(f"Extreme values detected (max abs: {max_val:.2e})")

        if issues:
            return False, "; ".join(issues)
        else:
            return True, "Data quality checks passed"

    except Exception as e:
        return False, f"Data quality validation failed: {e}"


def normalize_data(
    data: np.ndarray, method: str = "zscore", axis: Optional[int] = None
) -> np.ndarray:
    """
    Normalize data using various methods.

    This function provides different normalization methods for data preprocessing.

    Parameters
    ----------
    data : np.ndarray
        Input data to normalize.
    method : str, optional
        Normalization method: 'zscore', 'minmax', 'robust', or 'none'. Default is 'zscore'.
    axis : Optional[int], optional
        Axis along which to normalize. If None, normalize over all axes. Default is None.

    Returns
    -------
    np.ndarray
        Normalized data.

    Raises
    ------
    ValidationError
        If input parameters are invalid.
    ComputationError
        If normalization fails.

    Examples
    --------
    >>> import numpy as np
    >>> data = np.random.randn(2, 100, 10)
    >>> normalized = normalize_data(data, method='zscore', axis=1)
    >>> print(f"Original mean: {np.mean(data):.3f}")
    >>> print(f"Normalized mean: {np.mean(normalized):.3f}")
    """
    # Input validation
    if not isinstance(data, np.ndarray):
        raise ValidationError("data must be a numpy array", "data_type", type(data))

    if method not in ["zscore", "minmax", "robust", "none"]:
        raise ValidationError(
            "method must be one of: 'zscore', 'minmax', 'robust', 'none'",
            "method",
            method,
        )

    if axis is not None and not isinstance(axis, int):
        raise ValidationError(
            "axis must be an integer or None", "axis_type", type(axis)
        )

    try:
        if method == "none":
            return data.copy()

        if axis is None:
            data_flat = data.flatten()
        else:
            data_flat = data

        if method == "zscore":
            mean_val = np.mean(data_flat, axis=axis, keepdims=True)
            std_val = np.std(data_flat, axis=axis, keepdims=True)
            std_val = np.where(std_val == 0, 1, std_val)
            normalized = (data - mean_val) / std_val

        elif method == "minmax":
            min_val = np.min(data_flat, axis=axis, keepdims=True)
            max_val = np.max(data_flat, axis=axis, keepdims=True)
            range_val = max_val - min_val
            range_val = np.where(range_val == 0, 1, range_val)
            normalized = (data - min_val) / range_val

        elif method == "robust":
            median_val = np.median(data_flat, axis=axis, keepdims=True)
            mad_val = np.median(
                np.abs(data_flat - median_val), axis=axis, keepdims=True
            )
            mad_val = np.where(mad_val == 0, 1, mad_val)
            normalized = (data - median_val) / mad_val

        logger.info(f"Successfully normalized data using {method} method")
        return normalized

    except Exception as e:
        logger.error(f"Data normalization failed: {e}")
        raise ComputationError(
            f"Data normalization failed: {e}", "data_normalization", None
        )
