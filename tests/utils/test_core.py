import numpy as np
from numpy.testing import assert_allclose

from dcs.utils.core import (
    compute_conditional_event_statistics,
    compute_event_statistics,
    extract_event_snapshots,
    extract_event_windows,
)


def test_extract_event_windows():
    signal = np.arange(100)
    centers = np.array([20, 50, 80])
    result = extract_event_windows(signal, centers, start_offset=5, window_length=10)
    assert result.shape == (10, 3)
    assert_allclose(result[:, 0], np.arange(15, 25))


def test_extract_event_windows_out_of_bounds():
    signal = np.arange(50)
    centers = np.array([5, 48])
    try:
        extract_event_windows(signal, centers, 5, 10)
        assert False, "Expected IndexError"
    except IndexError:
        pass


def test_extract_event_snapshots():
    ts = np.random.randn(3, 100)
    locs = np.array([20, 40, 60])
    model_order = 1
    lag_step = 1
    result = extract_event_snapshots(ts, locs, model_order, lag_step, 2, 5)
    assert result.shape == (6, 5, 3)


def test_compute_event_statistics_output_shapes():
    data = np.random.randn(6, 10, 5)  # (nvar * (model_order + 1), time, trials)
    model_order = 1
    result = compute_event_statistics(data, model_order)
    assert "mean" in result and result["mean"].shape == (6, 10)
    assert "Sigma" in result and result["Sigma"].shape == (10, 6, 6)
    assert "OLS" in result and result["OLS"]["At"].shape == (10, 3, 3)


def test_compute_conditional_event_statistics_shapes():
    data = np.random.randn(6, 10, 5)  # (nvar * (model_order + 1), time, trials)
    model_order = 1
    result = compute_conditional_event_statistics(data, model_order)
    assert result["mean"].shape == (6, 10)
    assert result["Sigma"].shape == (10, 6, 6)
    assert result["OLS"]["At"].shape == (10, 3, 3)
    assert result["OLS"]["bt"].T.shape == (10, 3)
    assert result["OLS"]["Sigma_Et"].shape == (10, 3, 3)
    assert result["OLS"]["sigma_Et"].shape == (10, 1)
