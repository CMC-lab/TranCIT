from unittest.mock import patch

import numpy as np
import pytest
from dcs.config import (BicParams, CausalParams, DetectionParams, MonteCParams,
                        OutputParams, PipelineConfig, PipelineOptions)
from dcs.pipeline import snapshot_detect_analysis_pipeline


@pytest.fixture
def minimal_config():
    return PipelineConfig(
        options=PipelineOptions(
            detection=True,
            bic=False,
            causal_analysis=False,
            bootstrap=False,
            save_flag=False,
        ),
        detection=DetectionParams(
            thres_ratio=1.0,
            l_extract=10,
            l_start=5,
            align_type="peak",
            shrink_flag=False,
            remove_artif=False,
            locs=None,
        ),
        bic=BicParams(
            morder=1,
            momax=1,
            mode="OLS",
            tau=1,
        ),
        causal=CausalParams(
            ref_time=0,
            estim_mode="OLS",
            diag_flag=False,
            old_version=False,
        ),
        monte_carlo=MonteCParams(n_btsp=2),
        output=OutputParams(file_keyword="test_output")
    )

@patch("dcs.pipeline.find_peak_loc", return_value=np.array([30, 60]))
@patch("dcs.pipeline.extract_event_snapshots", return_value=np.random.randn(6, 10, 2))
@patch("dcs.pipeline.compute_event_statistics", return_value={
    "mean": np.random.randn(6, 10),
    "Sigma": np.random.randn(10, 6, 6),
    "OLS": {
        "At": np.random.randn(10, 2, 2),
        "bt": np.zeros((10, 2)),
        "Sigma_Et": np.array([np.eye(2) for _ in range(10)]),
        "sigma_Et": np.zeros((10, 2))
    }
})
def test_pipeline_basic(mock_stats, mock_snapshots, mock_peaks, minimal_config):
    orig_signal = np.random.randn(2, 100)
    detect_signal = np.random.randn(2, 100)

    result, _, snapshots = snapshot_detect_analysis_pipeline(
        orig_signal, detect_signal, minimal_config
    )

    assert isinstance(result, dict)
    assert "locs" in result
    assert "morder" in result
    assert "Yt_stats" in result
    assert snapshots.shape[0] == 6
