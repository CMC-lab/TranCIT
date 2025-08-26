import numpy as np
import pytest

from trancit.utils.residuals import estimate_residuals, get_residuals


def test_estimate_residuals_shapes():
    L, nvar, morder = 10, 2, 2
    At = np.random.randn(L, nvar, nvar * morder)
    Sigma = np.random.randn(L, nvar * (morder + 1), nvar * (morder + 1))
    Sigma = np.matmul(Sigma, np.transpose(Sigma, (0, 2, 1)))  # Ensure PSD
    mean = np.random.randn(nvar * (morder + 1), L)

    stats = {
        "OLS": {"At": At},
        "Sigma": Sigma,
        "mean": mean,
    }

    bt, Sigma_Et, sigma_Et = estimate_residuals(stats)
    assert bt.shape == (nvar, L)
    assert Sigma_Et.shape == (L, nvar, nvar)
    assert sigma_Et.shape == (L, 1)


def test_get_residuals_output():
    L, nvar, morder, ntrials = 8, 2, 2, 5
    event_data = np.random.randn(nvar * (morder + 1), L, ntrials)
    At = np.random.randn(L, nvar, nvar * morder)

    stats = {"OLS": {"At": At}}

    residuals = get_residuals(event_data, stats)
    assert residuals.shape == (nvar, L, ntrials)


def test_get_residuals_shape_mismatch_raises():
    L, nvar, morder, ntrials = 5, 2, 2, 3
    event_data = np.random.randn(nvar * (morder + 1), L, ntrials)
    At = np.random.randn(L, nvar, nvar * morder + 1)  # Wrong shape

    stats = {"OLS": {"At": At}}

    with pytest.raises(ValueError, match="Shape mismatch at time"):
        get_residuals(event_data, stats)
