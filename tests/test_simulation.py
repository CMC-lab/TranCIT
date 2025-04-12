import numpy as np
from dcs.simulation import (
    generate_ensemble_nonstat_innomean,
    generate_signals,
    simulate_ar_event,
    simulate_ar_event_bootstrap,
    simulate_ar_nonstat_innomean,
)


def test_simulate_ar_event_bootstrap_shape():
    simobj = {"nvar": 2, "morder": 2, "L": 10, "Ntrials": 5}
    Yt_event = np.random.randn(6, 10, 3)
    Yt_stats = {"OLS": {"At": np.random.randn(10, 2, 4)}}
    Et = np.random.randn(2, 10, 3)
    result = simulate_ar_event_bootstrap(simobj, Yt_event, Yt_stats, Et)
    assert result.shape == (6, 10, 5)


def test_simulate_ar_event_shape():
    simobj = {"nvar": 2, "morder": 2, "L": 10, "Ntrials": 3}
    Yt_stats = {
        "OLS": {
            "At": np.random.randn(10, 2, 4),
            "Sigma_Et": np.array([np.eye(2) for _ in range(10)]),
            "bt": np.zeros((2, 10)),
        },
        "mean": np.zeros((6, 10)),
        "Sigma": np.array([np.eye(6) for _ in range(10)]),
    }
    result = simulate_ar_event(simobj, Yt_stats)
    assert result.shape == (6, 10, 3)


def test_simulate_ar_nonstat_innomean_shape():
    A = np.random.randn(2, 4)
    SIG = np.eye(2)
    innomean = np.random.randn(2, 20)
    result = simulate_ar_nonstat_innomean(A, SIG, innomean, morder=2)
    assert result.shape == (2, 20)


def test_generate_signals_shapes():
    X, ns_x, ns_y = generate_signals(
        600, 3, 0.1, 0.05, 0.05, 1.0, 1.0, apply_morlet=False
    )
    assert X.shape == (2, 100, 3)
    assert ns_x.shape == (601,)
    assert ns_y.shape == (601,)


def test_generate_ensemble_nonstat_innomean_shape():
    A = np.random.randn(2, 4)
    SIG = np.eye(2)
    result, imp = generate_ensemble_nonstat_innomean(
        A, SIG, ntrials=3, L_event=100, center=50, amp=1.0, dim=1, L_perturb=100
    )
    assert result.shape == (2, 100, 3)
    assert imp.shape == (2, 100, 3)
