import numpy as np
import pytest
from dcs.models import estimate_var_coefficients
from dcs.simulation import generate_ensemble_nonstat_innomean


# --- Setup function for generating data (optional, makes tests cleaner) ---
def _generate_test_var_data(n_vars, morder, T, Ntrials, A_coeffs, SIG, amp=0):
    """Helper to generate VAR data for testing."""
    # Use dummy perturbation parameters if amp is 0
    L_perturb = 1 if amp == 0 else 51
    center = T // 2
    dim = 0
    data, _ = generate_ensemble_nonstat_innomean(
        A=A_coeffs,
        SIG=SIG,
        ntrials=Ntrials,
        L_event=T,
        center=center,
        amp=amp,
        dim=dim,
        L_perturb=L_perturb,
    )
    # Expected shape (n_vars, T, Ntrials)
    return data

# --- Tests for estimate_var_coefficients ---

def test_estimate_var_coefficients_recover_var1_params():
    """
    Test if VAR(1) coefficients can be recovered reasonably well.
    Uses time_mode='inhomo' and lag_mode='var'.
    """
    n_vars = 2
    morder = 1
    T = 500   # Sufficient time points
    Ntrials = 30 # More trials improve estimation
    SIG = np.eye(n_vars) * 0.01 # Low noise Cholesky factor

    # True coefficients A (shape n_vars x n_vars*morder = 2x2)
    A_true = np.array([[0.5, 0.3],
                       [0.0, 0.6]])

    # Generate data with known parameters
    data = _generate_test_var_data(n_vars, morder, T, Ntrials, A_true, SIG)

    # Estimate using the function
    # Note: max_model_order is needed; set it equal to morder when lag_mode='var'
    coeffs_est, omega_est, logL, sum_logdetH = estimate_var_coefficients(
        time_series_data=data,
        model_order=morder,
        max_model_order=morder, # Should equal model_order for lag_mode='var'
        time_mode='inhomo',
        lag_mode='var'
    )

    # Coefficients are time-varying in 'inhomo' mode. Average them for comparison.
    A_est_mean = np.mean(coeffs_est, axis=0)

    # Check if estimated A is close to true A (allow some tolerance)
    np.testing.assert_allclose(A_est_mean, A_true, atol=0.1)

    # Check residual covariance Omega (should be close to SIG @ SIG.T)
    omega_true = SIG @ SIG.T
    omega_est_mean = np.mean(omega_est, axis=0)
    np.testing.assert_allclose(omega_est_mean, omega_true, atol=0.05)

def test_estimate_var_coefficients_recover_var2_params():
    """
    Test if VAR(2) coefficients can be recovered reasonably well.
    Uses time_mode='inhomo' and lag_mode='var'.
    """
    n_vars = 2
    morder = 2
    T = 800   # Longer time for higher order
    Ntrials = 50 # More trials
    SIG = np.eye(n_vars) * 0.01

    # True coefficients A (shape n_vars x n_vars*morder = 2x4)
    A_true = np.array([[0.5, 0.1, 0.2, 0.0],  # A1 = [[0.5, 0.1], [0.0, 0.4]]
                       [0.0, 0.4, 0.0, 0.1]]) # A2 = [[0.2, 0.0], [0.0, 0.1]]

    data = _generate_test_var_data(n_vars, morder, T, Ntrials, A_true, SIG)

    coeffs_est, omega_est, logL, sum_logdetH = estimate_var_coefficients(
        time_series_data=data,
        model_order=morder,
        max_model_order=morder,
        time_mode='inhomo',
        lag_mode='var'
    )

    A_est_mean = np.mean(coeffs_est, axis=0)
    np.testing.assert_allclose(A_est_mean, A_true, atol=0.15) # Higher tolerance for VAR(2)

    omega_true = SIG @ SIG.T
    omega_est_mean = np.mean(omega_est, axis=0)
    np.testing.assert_allclose(omega_est_mean, omega_true, atol=0.05)

# --- Input Validation Tests ---

@pytest.fixture # Use a fixture to create common data for validation tests
def valid_data():
    n_vars=2; morder=1; T=50; Ntrials=5
    A_true = np.zeros((n_vars, n_vars*morder))
    SIG = np.eye(n_vars)
    return _generate_test_var_data(n_vars, morder, T, Ntrials, A_true, SIG)

def test_estimate_var_coeffs_invalid_data_dim(valid_data):
    """ Test error if data is not 3D. """
    with pytest.raises(ValueError, match="must be 3-dimensional"):
        estimate_var_coefficients(valid_data[:,:,0], 1, 1, 'inhomo', 'var') # Pass 2D data

def test_estimate_var_coeffs_invalid_nvars(valid_data):
    """ Test error if n_vars <= 1. """
    with pytest.raises(ValueError, match="n_vars > 1"):
        estimate_var_coefficients(valid_data[0:1,:,:], 1, 1, 'inhomo', 'var') # Pass 1 variable

def test_estimate_var_coeffs_invalid_nobs(valid_data):
    """ Test error if n_observations <= model_order. """
    morder = 50 # Order >= T (50)
    with pytest.raises(ValueError, match="Number of observations"):
         estimate_var_coefficients(valid_data, morder, morder, 'inhomo', 'var')

def test_estimate_var_coeffs_invalid_time_mode(valid_data):
    """ Test error for invalid time_mode. """
    with pytest.raises(ValueError, match="Invalid time_mode"):
        estimate_var_coefficients(valid_data, 1, 1, 'bad_mode', 'var')

def test_estimate_var_coeffs_invalid_lag_mode(valid_data):
    """ Test error for invalid lag_mode. """
    with pytest.raises(ValueError, match="Invalid lag_mode"):
        estimate_var_coefficients(valid_data, 1, 1, 'inhomo', 'bad_mode')
