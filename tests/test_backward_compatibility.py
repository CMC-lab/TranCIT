"""
Test backward compatibility with previous implementation.

This test ensures that the new implementation produces exactly
the same results as the previous implementation.
"""

import numpy as np
import pytest
from typing import Tuple, Dict, Any

# Import new implementation
from dcs import (
    compute_causal_strength_nonzero_mean,
    time_varying_causality,
    snapshot_detect_analysis_pipeline,
    DCSCalculator,
    PipelineOrchestrator,
    PipelineConfig,
    PipelineOptions,
    DetectionParams,
    BicParams,
    CausalParams,
    OutputParams,
    generate_signals
)

# Import previous implementation
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'prev'))
from prev.causality import compute_causal_strength_nonzero_mean as prev_compute_causal_strength_nonzero_mean
from prev.causality import time_varying_causality as prev_time_varying_causality
from prev.pipeline import snapshot_detect_analysis_pipeline as prev_snapshot_detect_analysis_pipeline


class TestBackwardCompatibility:
    """
    Test backward compatibility with previous implementation.
    
    This test suite ensures that all three main functions produce
    exactly the same results as the previous implementation.
    """
    
    @pytest.fixture
    def sample_data(self) -> np.ndarray:
        """Generate sample time series data for testing."""
        np.random.seed(42)
        return np.random.randn(2, 100, 10)  # (n_vars, n_obs, n_trials)
    
    @pytest.fixture
    def sample_event_data(self) -> np.ndarray:
        """Generate sample event data for testing."""
        np.random.seed(42)
        return np.random.randn(6, 50, 10)  # (nvar * (morder + 1), nobs, ntrials)
    
    @pytest.fixture
    def sample_stats(self) -> Dict[str, Any]:
        """Generate sample statistics for testing."""
        np.random.seed(42)
        return {
            "OLS": {
                "At": np.random.randn(50, 2, 4),
                "Sigma_Et": np.array([np.eye(2) for _ in range(50)])
            },
            "Sigma": np.random.randn(50, 6, 6),
            "mean": np.random.randn(6, 50)
        }
    
    @pytest.fixture
    def sample_causal_params(self) -> Dict[str, Any]:
        """Generate sample causal parameters for testing."""
        return {
            "ref_time": 10,
            "estim_mode": "OLS",
            "morder": 2,
            "diag_flag": False,
            "old_version": False
        }
    
    @pytest.fixture
    def sample_config(self) -> PipelineConfig:
        """Generate sample pipeline configuration for testing."""
        return PipelineConfig(
            options=PipelineOptions(
                detection=True,
                bic=False,
                causal_analysis=True,
                bootstrap=False,
                save_flag=False,
                debiased_stats=False
            ),
            detection=DetectionParams(
                thres_ratio=2.0,
                align_type="peak",
                l_extract=100,
                l_start=50,
                shrink_flag=False,
                locs=None,
                remove_artif=False
            ),
            bic=BicParams(
                morder=4,
                momax=None,
                tau=None,
                mode=None
            ),
            causal=CausalParams(
                ref_time=10,
                estim_mode="OLS",
                diag_flag=False,
                old_version=False
            ),
            output=OutputParams(
                file_keyword="test",
                save_path=""
            )
        )
    
    @pytest.fixture
    def sample_signals(self) -> Tuple[np.ndarray, np.ndarray]:
        """Generate sample original and detection signals for testing."""
        np.random.seed(42)
        original_signal = np.random.randn(2, 1000)
        detection_signal = np.random.randn(2, 1000)
        return original_signal, detection_signal

    def test_compute_causal_strength_nonzero_mean_identical(self, sample_data: np.ndarray) -> None:
        """Test that compute_causal_strength_nonzero_mean produces identical results."""
        model_order = 4
        time_mode = "inhomo"
        use_diagonal_covariance = False
        
        # Test new implementation
        new_result = compute_causal_strength_nonzero_mean(
            sample_data, model_order, time_mode, use_diagonal_covariance
        )
        
        # Test previous implementation
        prev_result = prev_compute_causal_strength_nonzero_mean(
            sample_data, model_order, time_mode, use_diagonal_covariance
        )
        
        # Compare all outputs
        for i, (new_val, prev_val) in enumerate(zip(new_result, prev_result)):
            np.testing.assert_array_almost_equal(new_val, prev_val, decimal=10)
    
    def test_compute_causal_strength_nonzero_mean_homogeneous(self, sample_data: np.ndarray) -> None:
        """Test homogeneous mode of compute_causal_strength_nonzero_mean."""
        model_order = 4
        time_mode = "homo"
        use_diagonal_covariance = False
        
        # Test new implementation
        new_result = compute_causal_strength_nonzero_mean(
            sample_data, model_order, time_mode, use_diagonal_covariance
        )
        
        # Test previous implementation
        prev_result = prev_compute_causal_strength_nonzero_mean(
            sample_data, model_order, time_mode, use_diagonal_covariance
        )
        
        # Compare all outputs
        for i, (new_val, prev_val) in enumerate(zip(new_result, prev_result)):
            np.testing.assert_array_almost_equal(new_val, prev_val, decimal=10)
    
    def test_compute_causal_strength_nonzero_mean_diagonal_covariance(self, sample_data: np.ndarray) -> None:
        """Test diagonal covariance mode of compute_causal_strength_nonzero_mean."""
        model_order = 4
        time_mode = "inhomo"
        use_diagonal_covariance = True
        
        # Test new implementation
        new_result = compute_causal_strength_nonzero_mean(
            sample_data, model_order, time_mode, use_diagonal_covariance
        )
        
        # Test previous implementation
        prev_result = prev_compute_causal_strength_nonzero_mean(
            sample_data, model_order, time_mode, use_diagonal_covariance
        )
        
        # Compare all outputs
        for i, (new_val, prev_val) in enumerate(zip(new_result, prev_result)):
            np.testing.assert_array_almost_equal(new_val, prev_val, decimal=10)
    
    def test_time_varying_causality_identical(self, sample_event_data: np.ndarray, 
                                            sample_stats: Dict[str, Any], 
                                            sample_causal_params: Dict[str, Any]) -> None:
        """Test that time_varying_causality produces identical results."""
        # Test new implementation
        new_result = time_varying_causality(sample_event_data, sample_stats, sample_causal_params)
        
        # Test previous implementation
        prev_result = prev_time_varying_causality(sample_event_data, sample_stats, sample_causal_params)
        
        # Compare all outputs
        for key in new_result.keys():
            np.testing.assert_array_almost_equal(new_result[key], prev_result[key], decimal=10)
    
    def test_time_varying_causality_old_version(self, sample_event_data: np.ndarray, 
                                              sample_stats: Dict[str, Any]) -> None:
        """Test old version of time_varying_causality."""
        causal_params = {
            "ref_time": 10,
            "estim_mode": "OLS",
            "morder": 2,
            "diag_flag": False,
            "old_version": True
        }
        
        # Test new implementation
        new_result = time_varying_causality(sample_event_data, sample_stats, causal_params)
        
        # Test previous implementation
        prev_result = prev_time_varying_causality(sample_event_data, sample_stats, causal_params)
        
        # Compare all outputs
        for key in new_result.keys():
            np.testing.assert_array_almost_equal(new_result[key], prev_result[key], decimal=10)
    
    def test_time_varying_causality_diagonal(self, sample_event_data: np.ndarray, 
                                           sample_stats: Dict[str, Any]) -> None:
        """Test diagonal covariance mode of time_varying_causality."""
        causal_params = {
            "ref_time": 10,
            "estim_mode": "OLS",
            "morder": 2,
            "diag_flag": True,
            "old_version": False
        }
        
        # Test new implementation
        new_result = time_varying_causality(sample_event_data, sample_stats, causal_params)
        
        # Test previous implementation
        prev_result = prev_time_varying_causality(sample_event_data, sample_stats, causal_params)
        
        # Compare all outputs
        for key in new_result.keys():
            np.testing.assert_array_almost_equal(new_result[key], prev_result[key], decimal=10)
    
    def test_snapshot_detect_analysis_pipeline_identical(self, sample_signals: Tuple[np.ndarray, np.ndarray], 
                                                        sample_config: PipelineConfig) -> None:
        """Test that snapshot_detect_analysis_pipeline produces identical results."""
        original_signal, detection_signal = sample_signals
        
        # Test new implementation
        new_results, new_config, new_snapshots = snapshot_detect_analysis_pipeline(
            original_signal, detection_signal, sample_config
        )
        
        # Test previous implementation
        prev_results, prev_config, prev_snapshots = prev_snapshot_detect_analysis_pipeline(
            original_signal, detection_signal, sample_config
        )
        
        # Compare results structure
        assert set(new_results.keys()) == set(prev_results.keys())
        
        # Compare key outputs
        np.testing.assert_array_equal(new_results["locs"], prev_results["locs"])
        assert new_results["morder"] == prev_results["morder"]
        
        # Compare snapshots if they exist
        if new_snapshots.size > 0 and prev_snapshots.size > 0:
            np.testing.assert_array_almost_equal(new_snapshots, prev_snapshots, decimal=10)
    
    def test_dcs_calculator_identical(self, sample_data: np.ndarray) -> None:
        """Test that DCSCalculator produces identical results to legacy function."""
        model_order = 4
        time_mode = "inhomo"
        use_diagonal_covariance = False
        
        # Test DCSCalculator
        calculator = DCSCalculator(
            model_order=model_order,
            time_mode=time_mode,
            use_diagonal_covariance=use_diagonal_covariance
        )
        calculator_result = calculator.analyze(sample_data)
        
        # Test legacy function
        legacy_result = compute_causal_strength_nonzero_mean(
            sample_data, model_order, time_mode, use_diagonal_covariance
        )
        
        # Compare outputs
        np.testing.assert_array_almost_equal(calculator_result.causal_strength, legacy_result[0], decimal=10)
        np.testing.assert_array_almost_equal(calculator_result.transfer_entropy, legacy_result[1], decimal=10)
        np.testing.assert_array_almost_equal(calculator_result.granger_causality, legacy_result[2], decimal=10)
        np.testing.assert_array_almost_equal(calculator_result.coefficients, legacy_result[3], decimal=10)
        np.testing.assert_array_almost_equal(calculator_result.te_residual_cov, legacy_result[4], decimal=10)
    
    def test_pipeline_compatibility(self, sample_signals: Tuple[np.ndarray, np.ndarray], 
                                  sample_config: PipelineConfig) -> None:
        """Test that PipelineOrchestrator produces compatible results."""
        original_signal, detection_signal = sample_signals
        
        # Test PipelineOrchestrator
        orchestrator = PipelineOrchestrator(sample_config)
        orchestrator_result = orchestrator.run(original_signal, detection_signal)
        
        # Test legacy function
        legacy_results, legacy_config, legacy_snapshots = snapshot_detect_analysis_pipeline(
            original_signal, detection_signal, sample_config
        )
        
        # Compare key outputs
        np.testing.assert_array_equal(orchestrator_result.results["locs"], legacy_results["locs"])
        assert orchestrator_result.results["morder"] == legacy_results["morder"]
        
        # Compare snapshots if they exist
        if orchestrator_result.event_snapshots.size > 0 and legacy_snapshots.size > 0:
            np.testing.assert_array_almost_equal(orchestrator_result.event_snapshots, legacy_snapshots, decimal=10)
    
    def test_import_compatibility(self) -> None:
        """Test that all legacy and new classes/functions can be imported without errors."""
        # Test legacy function imports
        from dcs import (
            compute_causal_strength_nonzero_mean,
            time_varying_causality,
            snapshot_detect_analysis_pipeline
        )
        
        # Test new class imports
        from dcs import (
            DCSCalculator,
            PipelineOrchestrator,
            PipelineConfig
        )
        
        # Test utility imports
        from dcs import (
            compute_event_statistics,
            extract_event_snapshots,
            find_peak_locations
        )
        
        # Test model imports
        from dcs import (
            VAREstimator,
            BICSelector,
            compute_multi_trial_BIC
        )
        
        assert True  # If we get here, all imports work


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v"]) 
