import logging
from typing import Tuple

import numpy as np
from sklearn.covariance import ledoit_wolf

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

    Notes
    -----
    - The function examines only the first two variables (rows) of `event_data`.
    - Trials are removed based on the condition `event_data[:2, :, :] < lower_threshold`.
    """
    # Find indices where any value in the first two variables is below the threshold
    artifact_indices = np.where(event_data[:2, :, :] < lower_threshold)

    # Get unique trial indices to remove
    trials_to_remove = np.unique(artifact_indices[2])

    # Remove the artifact trials from event_data and locations
    updated_event_data = np.delete(event_data, trials_to_remove, axis=2)
    updated_locations = np.delete(locations, trials_to_remove)

    # Log the number of removed trials
    print(f"Removed {len(trials_to_remove)} artifact trials")

    return updated_event_data, updated_locations


def regularize_if_singular(
    matrix: np.ndarray, samples=None, epsilon: float = 1e-4, threshold: float = 1e-6
) -> np.ndarray:
    """
    Check if a matrix is singular and regularize it by adding epsilon to the diagonal if needed.

    This function checks if the determinant of the matrix is below a specified threshold.
    If it is, the matrix is considered singular, and a small value (epsilon) is added to its diagonal
    to make it invertible. Otherwise, the original matrix is returned.

    Parameters
    ----------
    matrix : np.ndarray
        Square matrix to check and potentially regularize.
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
    ValueError
        If the input matrix is not square.

    Notes
    -----
    - The function uses the absolute value of the determinant to check for singularity.
    - Uncommented code provides an alternative approach using condition number.
    """
    if matrix.shape[0] != matrix.shape[1]:
        raise ValueError("Input matrix must be square")

    # cond_number = np.linalg.cond(matrix)
    # if cond_number > threshold:
    #     regularized_matrix = matrix + epsilon * np.eye(matrix.shape[0])
    #     return regularized_matrix
    # else:
    #     return matrix
    # det = np.linalg.det(matrix)
    max_iter = 2
    
    det = np.linalg.det(matrix)
    if abs(det) < threshold:
        logger.warning(f"Singular matrix (det={det:.2e})")

        # 1) Try one-shot Ledoit–Wolf shrinkage if we have samples
        if samples is not None:
            matrix, alpha = ledoit_wolf(samples)
            logger.info(f"Ledoit-Wolf alpha={alpha:.3f}")
            return matrix

        # 2) Iteratively add ridge until no longer singular
        ridge_factor = epsilon
        for i in range(1, max_iter + 1):
            scale = np.mean(np.diag(matrix))
            matrix = matrix + ridge_factor * scale * np.eye(matrix.shape[0])
            ridge_factor *= 2
    return matrix
