import numpy as np
from dcs.causality import compute_causal_strength_nonzero_mean


def test_compute_causal_strength_nonzero_mean_shapes():
    np.random.seed(0)
    data = np.random.randn(2, 50, 10)
    model_order = 2

    cs, te, gc, coeffs, te_cov = compute_causal_strength_nonzero_mean(
        data, model_order=model_order, time_mode="inhomo", use_diagonal_covariance=False
    )

    assert cs.shape[1] == 2
    assert te.shape[1] == 2
    assert gc.shape[1] == 2
    assert coeffs.shape == (50, 2, 2 * model_order + 1)
    assert te_cov.shape[1] == 2
