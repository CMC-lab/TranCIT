import numpy as np
import pytest
from dcs.utils.preprocess import regularize_if_singular, remove_artifact_trials


def test_regularize_if_singular_non_singular_matrix():
    """
    Test that a non-singular matrix is returned unchanged.
    """
    non_singular_matrix = np.array([[2.0, 1.0], [1.0, 2.0]])
    # Determinant is 4 - 1 = 3, which is > threshold (default 1e-6)
    result = regularize_if_singular(non_singular_matrix)
    np.testing.assert_array_equal(result, non_singular_matrix)

def test_regularize_if_singular_singular_matrix():
    """
    Test that a singular matrix is regularized by adding epsilon to the diagonal.
    """
    singular_matrix = np.array([[1.0, 1.0], [1.0, 1.0]])
    # Determinant is 1 - 1 = 0, which is < threshold
    epsilon = 1e-6
    expected_regularized = np.array([[1.0 + epsilon, 1.0], [1.0, 1.0 + epsilon]])
    result = regularize_if_singular(singular_matrix, epsilon=epsilon)
    np.testing.assert_allclose(result, expected_regularized)

def test_regularize_if_singular_near_singular_matrix():
    """
    Test that a near-singular matrix (below threshold) is regularized.
    """
    near_singular_matrix = np.array([[1.0, 1.0], [1.0, 1.0 + 1e-7]])
    # Determinant is 1e-7, which is < threshold (default 1e-6)
    epsilon = 1e-6
    expected_regularized = near_singular_matrix + epsilon * np.eye(2)
    result = regularize_if_singular(near_singular_matrix, epsilon=epsilon, threshold=1e-6)
    np.testing.assert_allclose(result, expected_regularized)

def test_regularize_if_singular_non_square_matrix():
    """
    Test that a non-square matrix raises a ValueError.
    """
    non_square_matrix = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    with pytest.raises(ValueError, match="Input matrix must be square"):
        regularize_if_singular(non_square_matrix)


# --- Tests for remove_artifact_trials ---

def test_remove_artifact_no_artifacts():
    """
    Test case where no trials should be removed.
    """
    # 3 vars, 10 time points, 4 trials
    event_data = np.random.rand(3, 10, 4) + 0.1 # Ensure all values > 0.1
    locations = np.arange(4) * 100
    lower_threshold = 0.05

    updated_data, updated_locations = remove_artifact_trials(
        event_data, locations, lower_threshold
    )

    # Check shapes and content (should be unchanged)
    assert updated_data.shape == (3, 10, 4)
    np.testing.assert_array_equal(updated_data, event_data)
    assert updated_locations.shape == (4,)
    np.testing.assert_array_equal(updated_locations, locations)

def test_remove_artifact_single_trial_artifact_var0():
    """
    Test removal of a single trial due to artifact in variable 0.
    """
    event_data = np.ones((3, 10, 4)) * 0.5 # Base data above threshold
    event_data[0, 5, 2] = 0.01 # Artifact in var 0, trial 2
    locations = np.array([100, 200, 300, 400])
    lower_threshold = 0.1

    # Expected data after removing trial 2 (index 2)
    expected_data = np.delete(event_data, 2, axis=2)
    expected_locations = np.array([100, 200, 400])

    updated_data, updated_locations = remove_artifact_trials(
        event_data, locations, lower_threshold
    )

    # Check shapes and content
    assert updated_data.shape == (3, 10, 3)
    np.testing.assert_array_equal(updated_data, expected_data)
    assert updated_locations.shape == (3,)
    np.testing.assert_array_equal(updated_locations, expected_locations)

def test_remove_artifact_single_trial_artifact_var1():
    """
    Test removal of a single trial due to artifact in variable 1.
    """
    event_data = np.ones((3, 10, 4)) * 0.5 # Base data above threshold
    event_data[1, 2, 1] = 0.01 # Artifact in var 1, trial 1
    locations = np.array([100, 200, 300, 400])
    lower_threshold = 0.1

    # Expected data after removing trial 1 (index 1)
    expected_data = np.delete(event_data, 1, axis=2)
    expected_locations = np.array([100, 300, 400]) # Remove 200

    updated_data, updated_locations = remove_artifact_trials(
        event_data, locations, lower_threshold
    )

    # Check shapes and content
    assert updated_data.shape == (3, 10, 3)
    np.testing.assert_array_equal(updated_data, expected_data)
    assert updated_locations.shape == (3,)
    np.testing.assert_array_equal(updated_locations, expected_locations)

def test_remove_artifact_multiple_trials():
    """
    Test removal of multiple trials.
    """
    event_data = np.ones((3, 10, 5)) * 0.5 # Base data above threshold
    event_data[0, 5, 1] = 0.01 # Artifact in trial 1
    event_data[1, 8, 3] = 0.02 # Artifact in trial 3
    locations = np.array([10, 20, 30, 40, 50])
    lower_threshold = 0.1

    # Expected data after removing trials 1 and 3 (indices 1, 3)
    expected_data = np.delete(event_data, [1, 3], axis=2)
    expected_locations = np.array([10, 30, 50]) # Remove 20, 40

    updated_data, updated_locations = remove_artifact_trials(
        event_data, locations, lower_threshold
    )

    # Check shapes and content
    assert updated_data.shape == (3, 10, 3)
    np.testing.assert_array_equal(updated_data, expected_data)
    assert updated_locations.shape == (3,)
    np.testing.assert_array_equal(updated_locations, expected_locations)

def test_remove_artifact_value_at_threshold():
    """
    Test that values exactly at the threshold are NOT removed.
    """
    event_data = np.ones((3, 10, 4)) * 0.5
    event_data[0, 5, 2] = 0.1 # Value exactly at threshold
    locations = np.arange(4) * 100
    lower_threshold = 0.1

    updated_data, updated_locations = remove_artifact_trials(
        event_data, locations, lower_threshold
    )

    # Should be unchanged as value is not strictly < threshold
    assert updated_data.shape == (3, 10, 4)
    np.testing.assert_array_equal(updated_data, event_data)
    assert updated_locations.shape == (4,)
    np.testing.assert_array_equal(updated_locations, locations)

def test_remove_artifact_irrelevant_variable():
    """
    Test that artifacts in variables other than the first two are ignored.
    """
    event_data = np.ones((3, 10, 4)) * 0.5
    event_data[2, 5, 2] = 0.01 # Artifact in var 2 (should be ignored)
    locations = np.arange(4) * 100
    lower_threshold = 0.1

    updated_data, updated_locations = remove_artifact_trials(
        event_data, locations, lower_threshold
    )

    # Should be unchanged
    assert updated_data.shape == (3, 10, 4)
    np.testing.assert_array_equal(updated_data, event_data)
    assert updated_locations.shape == (4,)
    np.testing.assert_array_equal(updated_locations, locations)

# --- Edge Case Tests ---

def test_remove_artifact_empty_trials():
    """
    Test behavior with zero trials.
    """
    event_data = np.empty((3, 10, 0))
    locations = np.empty((0,))
    lower_threshold = 0.1

    updated_data, updated_locations = remove_artifact_trials(
        event_data, locations, lower_threshold
    )

    assert updated_data.shape == (3, 10, 0)
    assert updated_locations.shape == (0,)
