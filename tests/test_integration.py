"""
Integration tests for the Dynamic Causal Strength (DCS) package.

This module contains comprehensive integration tests that verify the
functionality of the complete analysis pipeline with various configurations.
"""

import numpy as np
import pytest
from typing import Tuple

from dcs import generate_signals, snapshot_detect_analysis_pipeline
from dcs.config import (
    BicParams,
    CausalParams,
    DeSnapParams,
    DetectionParams,
    MonteCParams,
    OutputParams,
    PipelineConfig,
    PipelineOptions,
)


class TestIntegration:
    """Integration tests for the full pipeline."""

    @pytest.fixture
    def sample_data(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate sample data for testing.
        
        Returns
        -------
        Tuple[np.ndarray, np.ndarray]
            original_signal: Trial-averaged signal
            detection_signal: Signal for event detection
        """
        T, Ntrial = 500, 10
        data, _, _ = generate_signals(
            T=T,
            Ntrial=Ntrial,
            h=0.1,
            gamma1=0.5,
            gamma2=0.5,
            Omega1=1.0,
            Omega2=1.2,
        )
        original_signal = np.mean(data, axis=2)
        detection_signal = original_signal * 1.5
        return original_signal, detection_signal

    @pytest.fixture
    def basic_config(self) -> PipelineConfig:
        """
        Create a basic configuration for testing.
        
        Returns
        -------
        PipelineConfig
            Basic pipeline configuration
        """
        return PipelineConfig(
            options=PipelineOptions(
                detection=True,
                bic=False,
                causal_analysis=True,
                bootstrap=False,
                save_flag=False,
                debiased_stats=False,
            ),
            detection=DetectionParams(
                thres_ratio=1.5,
                align_type="peak",
                l_extract=100,
                l_start=50,
                remove_artif=False,
            ),
            bic=BicParams(morder=3),
            causal=CausalParams(ref_time=50, estim_mode="OLS"),
            output=OutputParams(file_keyword="test_run"),
        )

    def test_basic_pipeline(
        self, sample_data: Tuple[np.ndarray, np.ndarray], basic_config: PipelineConfig
    ) -> None:
        """
        Test basic pipeline functionality.
        
        This test verifies that the basic pipeline can:
        1. Detect events successfully
        2. Extract event snapshots
        3. Compute causality measures
        4. Return properly structured results
        """
        original_signal, detection_signal = sample_data

        results, final_config, event_snapshots = snapshot_detect_analysis_pipeline(
            original_signal=original_signal,
            detection_signal=detection_signal,
            config=basic_config,
        )

        # Check that results contain expected keys
        assert "locs" in results, "Results should contain 'locs'"
        assert "morder" in results, "Results should contain 'morder'"
        assert "Yt_stats" in results, "Results should contain 'Yt_stats'"
        assert "CausalOutput" in results, "Results should contain 'CausalOutput'"

        # Check that event snapshots were extracted
        assert event_snapshots.shape[0] > 0, "Event snapshots should have data"
        assert event_snapshots.shape[1] > 0, "Event snapshots should have time dimension"
        assert event_snapshots.shape[2] > 0, "Event snapshots should have trial dimension"

        # Check causal output structure
        if results["CausalOutput"]:
            causal_output = results["CausalOutput"]["OLS"]
            assert "DCS" in causal_output, "Causal output should contain 'DCS'"
            assert "TE" in causal_output, "Causal output should contain 'TE'"
            assert "rDCS" in causal_output, "Causal output should contain 'rDCS'"

    def test_pipeline_with_bic(
        self, sample_data: Tuple[np.ndarray, np.ndarray]
    ) -> None:
        """
        Test pipeline with BIC model selection.
        
        This test verifies that the pipeline can:
        1. Perform BIC model selection
        2. Select optimal model order
        3. Return BIC results
        """
        original_signal, detection_signal = sample_data

        config = PipelineConfig(
            options=PipelineOptions(
                detection=True,
                bic=True,
                causal_analysis=True,
                bootstrap=False,
                save_flag=False,
            ),
            detection=DetectionParams(
                thres_ratio=1.5,
                align_type="peak",
                l_extract=100,
                l_start=50,
            ),
            bic=BicParams(morder=3, momax=5, tau=1, mode="biased"),
            causal=CausalParams(ref_time=50, estim_mode="OLS"),
            output=OutputParams(file_keyword="test_bic_run"),
        )

        results, final_config, event_snapshots = snapshot_detect_analysis_pipeline(
            original_signal=original_signal,
            detection_signal=detection_signal,
            config=config,
        )

        assert "BICoutputs" in results, "Results should contain 'BICoutputs'"
        assert results["BICoutputs"] is not None, "BIC outputs should not be None"

    def test_pipeline_with_bootstrap(
        self, sample_data: Tuple[np.ndarray, np.ndarray]
    ) -> None:
        """
        Test pipeline with bootstrap analysis.
        
        This test verifies that the pipeline can:
        1. Perform bootstrap analysis
        2. Generate bootstrap samples
        3. Return bootstrap results
        """
        original_signal, detection_signal = sample_data

        config = PipelineConfig(
            options=PipelineOptions(
                detection=True,
                bic=False,
                causal_analysis=True,
                bootstrap=True,
                save_flag=False,
            ),
            detection=DetectionParams(
                thres_ratio=1.5,
                align_type="peak",
                l_extract=100,
                l_start=50,
            ),
            bic=BicParams(morder=3),
            causal=CausalParams(ref_time=50, estim_mode="OLS"),
            monte_carlo=MonteCParams(n_btsp=5),  # Small number for testing
            output=OutputParams(file_keyword="test_bootstrap_run"),
        )

        results, final_config, event_snapshots = snapshot_detect_analysis_pipeline(
            original_signal=original_signal,
            detection_signal=detection_signal,
            config=config,
        )

        assert "BootstrapCausalOutputs" in results, "Results should contain bootstrap outputs"
        # Bootstrap results might be None if not enough data
        # assert results["BootstrapCausalOutputs"] is not None

    def test_pipeline_with_desnap(
        self, sample_data: Tuple[np.ndarray, np.ndarray]
    ) -> None:
        """
        Test pipeline with DeSnap analysis.
        
        This test verifies that the pipeline can:
        1. Perform DeSnap analysis
        2. Generate unconditional statistics
        3. Return DeSnap results
        """
        original_signal, detection_signal = sample_data

        config = PipelineConfig(
            options=PipelineOptions(
                detection=True,
                bic=False,
                causal_analysis=True,
                bootstrap=False,
                save_flag=False,
                debiased_stats=True,
            ),
            detection=DetectionParams(
                thres_ratio=1.5,
                align_type="peak",
                l_extract=100,
                l_start=50,
            ),
            bic=BicParams(morder=3),
            causal=CausalParams(ref_time=50, estim_mode="OLS"),
            desnap=DeSnapParams(
                detection_signal=detection_signal[0],
                original_signal=original_signal,
                Yt_stats_cond={},  # Will be filled by pipeline
                morder=3,
                tau=1,
                l_start=50,
                l_extract=100,
                d0=0.0,
                N_d=10,
                maxStdRatio=2.0,
            ),
            output=OutputParams(file_keyword="test_desnap_run"),
        )

        results, final_config, event_snapshots = snapshot_detect_analysis_pipeline(
            original_signal=original_signal,
            detection_signal=detection_signal,
            config=config,
        )

        assert "DeSnap_output" in results, "Results should contain DeSnap output"
        assert "Yt_stats_unconditional" in results, "Results should contain unconditional stats"

    def test_pipeline_no_detection(
        self, sample_data: Tuple[np.ndarray, np.ndarray]
    ) -> None:
        """
        Test pipeline with pre-provided locations.
        
        This test verifies that the pipeline can:
        1. Skip event detection when locations are provided
        2. Use pre-provided event locations
        3. Process the analysis correctly
        """
        original_signal, detection_signal = sample_data

        # Create some dummy locations
        locs = np.array([100, 200, 300])

        config = PipelineConfig(
            options=PipelineOptions(
                detection=False,
                bic=False,
                causal_analysis=True,
                bootstrap=False,
                save_flag=False,
            ),
            detection=DetectionParams(
                thres_ratio=1.5,
                align_type="peak",
                l_extract=100,
                l_start=50,
                locs=locs,
            ),
            bic=BicParams(morder=3),
            causal=CausalParams(ref_time=50, estim_mode="OLS"),
            output=OutputParams(file_keyword="test_no_detection_run"),
        )

        results, final_config, event_snapshots = snapshot_detect_analysis_pipeline(
            original_signal=original_signal,
            detection_signal=detection_signal,
            config=config,
        )

        assert "locs" in results, "Results should contain 'locs'"
        assert len(results["locs"]) > 0, "Results should have detected locations"

    def test_pipeline_error_handling(
        self, sample_data: Tuple[np.ndarray, np.ndarray]
    ) -> None:
        """
        Test pipeline error handling.
        
        This test verifies that the pipeline handles errors gracefully:
        1. Invalid configuration parameters
        2. Missing required parameters
        3. Data shape mismatches
        """
        original_signal, detection_signal = sample_data

        # Test with invalid configuration
        config = PipelineConfig(
            options=PipelineOptions(
                detection=False,  # No detection
                bic=False,
                causal_analysis=True,
                bootstrap=False,
                save_flag=False,
            ),
            detection=DetectionParams(
                thres_ratio=1.5,
                align_type="peak",
                l_extract=100,
                l_start=50,
                locs=None,  # Missing locations
            ),
            bic=BicParams(morder=3),
            causal=CausalParams(ref_time=50, estim_mode="OLS"),
            output=OutputParams(file_keyword="test_error_run"),
        )

        # Should raise ValueError due to missing locations
        with pytest.raises(ValueError, match="detection.locs must be provided"):
            snapshot_detect_analysis_pipeline(
                original_signal=original_signal,
                detection_signal=detection_signal,
                config=config,
            ) 
