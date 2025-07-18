import numpy as np
from dcs.utils.helpers import compute_covariances, estimate_coefficients


def test_compute_covariances_shapes():
    nvars, morder, tsteps, trials = 2, 3, 5, 10
    lagged_data = np.random.randn(nvars, morder, tsteps, trials)

    cov_Xp, cov_Yp, C_XYp, C_YXp = compute_covariances(lagged_data, tsteps, morder)

    assert cov_Xp.shape == (tsteps, morder, morder)
    assert cov_Yp.shape == (tsteps, morder, morder)
    assert C_XYp.shape == (tsteps, morder, morder)
    assert C_YXp.shape == (tsteps, morder, morder)


def test_estimate_coefficients_shapes_and_validity():
    nvar, morder, ntrials = 2, 2, 50
    current_data = np.random.randn(ntrials, nvar)
    lagged_data = np.random.randn(ntrials, nvar * morder)

    coeff, res_cov = estimate_coefficients(current_data, lagged_data, ntrials)

    assert coeff.shape == (nvar, nvar * morder + 1)
    assert res_cov.shape == (nvar, nvar)
    assert np.all(np.isfinite(coeff))
    assert np.all(np.isfinite(res_cov))
