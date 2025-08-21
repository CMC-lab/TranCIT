"""
Test cases for pipeline functionality.

This module tests the refactored class-based pipeline API including
PipelineOrchestrator and individual pipeline stages.
"""

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from dcs.config import (
    BicParams,
    CausalParams,
    DetectionParams,
    MonteCParams,
    OutputParams,
    PipelineConfig,
    PipelineOptions,
)
from dcs.core.exceptions import ComputationError, ValidationError
from dcs.pipeline import (
    BICSelectionStage,
    CausalityAnalysisStage,
    EventDetectionStage,
    InputValidationStage,
    PipelineOrchestrator,
    PipelineResult,
    StatisticsComputationStage,
)


@pytest.fixture
def minimal_config():
    """Create minimal pipeline configuration for testing."""
    return PipelineConfig(
        options=PipelineOptions(
            detection=True,
            bic=False,
            causal_analysis=False,
            bootstrap=False,
            save_flag=False,
            debiased_stats=False,
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
        output=OutputParams(file_keyword="test_output"),
    )


@pytest.fixture
def full_config():
    """Create full pipeline configuration for testing."""
    return PipelineConfig(
        options=PipelineOptions(
            detection=True,
            bic=True,
            causal_analysis=True,
            bootstrap=False,
            save_flag=False,
            debiased_stats=False,
        ),
        detection=DetectionParams(
            thres_ratio=2.0,
            l_extract=50,
            l_start=25,
            align_type="peak",
            shrink_flag=False,
            remove_artif=True,
        ),
        bic=BicParams(
            morder=4,
            momax=6,
            mode="OLS",
            tau=1,
        ),
        causal=CausalParams(
            ref_time=25,
            estim_mode="OLS",
            diag_flag=False,
            old_version=False,
        ),
        monte_carlo=MonteCParams(n_btsp=10),
        output=OutputParams(file_keyword="full_test"),
    )


@pytest.fixture
def sample_signals():
    """Generate sample signals for testing."""
    np.random.seed(42)
    original_signal = np.random.randn(2, 200)
    detection_signal = original_signal * 1.5 + np.random.randn(2, 200) * 0.1
    return original_signal, detection_signal


class TestPipelineOrchestrator:
    """Test cases for PipelineOrchestrator class."""

    def test_initialization(self, minimal_config):
        """Test PipelineOrchestrator initialization."""
        orchestrator = PipelineOrchestrator(minimal_config)

        assert orchestrator.config == minimal_config
        assert hasattr(orchestrator, "stages")
        assert len(orchestrator.stages) > 0

        # Check that all expected stages are initialized
        expected_stages = [
            "input_validation",
            "event_detection",
            "border_removal",
            "bic_selection",
            "snapshot_extraction",
            "artifact_removal",
            "statistics_computation",
            "causality_analysis",
            "bootstrap_analysis",
            "desnap_analysis",
            "output_preparation",
        ]

        for stage_name in expected_stages:
            assert stage_name in orchestrator.stages

    def test_analyze_method(self, sample_signals, minimal_config):
        """Test analyze method (BaseAnalyzer interface compatibility)."""
        original_signal, detection_signal = sample_signals
        orchestrator = PipelineOrchestrator(minimal_config)

        result = orchestrator.analyze(
            original_signal, detection_signal=detection_signal
        )

        assert isinstance(result, PipelineResult)
        assert hasattr(result, "results")
        assert hasattr(result, "config")
        assert hasattr(result, "event_snapshots")

    @patch("dcs.pipeline.stages.EventDetectionStage")
    def test_run_method_with_mocked_detection(
        self, mock_detection, sample_signals, minimal_config
    ):
        """Test run method with mocked event detection."""
        original_signal, detection_signal = sample_signals

        # Mock the event detection to return some locations
        mock_stage = MagicMock()
        mock_stage.execute.return_value = {"locs": np.array([50, 100, 150])}
        mock_detection.return_value = mock_stage

        orchestrator = PipelineOrchestrator(minimal_config)
        result = orchestrator.run(original_signal, detection_signal)

        assert isinstance(result, PipelineResult)
        assert hasattr(result, "results")
        assert result.config == minimal_config

    def test_input_validation(self, minimal_config):
        """Test input validation."""
        orchestrator = PipelineOrchestrator(minimal_config)

        # Test with wrong dimensions
        with pytest.raises(ValidationError, match="Input signals must be 2D"):
            orchestrator.run(np.random.randn(2, 100, 5), np.random.randn(2, 100))

        # Test with wrong number of variables
        with pytest.raises(ValidationError, match="Input signals must be bivariate"):
            orchestrator.run(np.random.randn(3, 100), np.random.randn(3, 100))

    def test_missing_detection_signal_in_analyze(self, minimal_config):
        """Test analyze method without detection_signal in kwargs."""
        orchestrator = PipelineOrchestrator(minimal_config)

        with pytest.raises(ValueError, match="detection_signal must be provided"):
            orchestrator.analyze(np.random.randn(2, 100))


class TestPipelineStages:
    """Test cases for individual pipeline stages."""

    def test_input_validation_stage(self, minimal_config, sample_signals):
        """Test InputValidationStage."""
        original_signal, detection_signal = sample_signals
        stage = InputValidationStage(minimal_config)

        result = stage.execute(
            original_signal=original_signal, detection_signal=detection_signal
        )

        assert "original_signal" in result
        assert "detection_signal" in result
        assert np.array_equal(result["original_signal"], original_signal)
        assert np.array_equal(result["detection_signal"], detection_signal)

    def test_event_detection_stage_basic(self, minimal_config, sample_signals):
        """Test EventDetectionStage basic functionality."""
        original_signal, detection_signal = sample_signals
        stage = EventDetectionStage(minimal_config)

        # This might fail without proper signal characteristics, so we'll mock it
        with patch("dcs.utils.signal.find_peak_locations") as mock_find_peaks:
            mock_find_peaks.return_value = np.array([25, 75, 125])

            result = stage.execute(
                original_signal=original_signal, detection_signal=detection_signal
            )

            assert "locs" in result
            assert len(result["locs"]) > 0

    def test_bic_selection_stage(self, full_config, sample_signals):
        """Test BICSelectionStage."""
        original_signal, detection_signal = sample_signals
        stage = BICSelectionStage(full_config)

        # Mock the BIC computation since it requires proper data setup
        with patch("dcs.models.bic_selection.BICSelector") as mock_selector:
            mock_bic = MagicMock()
            mock_bic._compute_multi_trial_bic.return_value = {
                "BIC": np.array([10.0, 8.0, 12.0, 15.0]),
                "mobic": [1.0, 2.0],  # BIC expects [index, selected_model_order]
            }
            mock_selector.return_value = mock_bic

            result = stage.execute(
                original_signal=original_signal, locs=np.array([50, 100, 150])
            )

            assert "bic_outputs" in result
            assert "morder" in result

    def test_statistics_computation_stage(self, full_config):
        """Test StatisticsComputationStage."""
        stage = StatisticsComputationStage(full_config)

        # Create mock event snapshots
        # With model_order=4 and n_vars=2, first dim should be
        # n_vars*(model_order+1) = 2*(4+1) = 10
        event_snapshots = np.random.randn(
            10, 50, 20
        )  # (n_vars*(morder+1), n_obs, n_trials)

        with patch("dcs.utils.core.compute_event_statistics") as mock_stats:
            mock_stats.return_value = {
                "OLS": {
                    "At": np.random.randn(50, 2, 5),
                    "Sigma_Et": np.random.randn(50, 2, 2),
                },
                "Sigma": np.random.randn(50, 6, 6),
                "mean": np.random.randn(6, 50),
            }

            result = stage.execute(event_snapshots=event_snapshots, morder=4)

            assert "event_stats" in result

    def test_causality_analysis_stage(self, full_config):
        """Test CausalityAnalysisStage."""
        stage = CausalityAnalysisStage(full_config)

        # Create mock inputs
        event_data = np.random.randn(
            10, 50, 20
        )  # (nvar * (morder + 1), n_obs, n_trials)
        event_stats = {
            "OLS": {
                "At": np.random.randn(50, 2, 5),
                "Sigma_Et": np.random.randn(50, 2, 2),
            },
            "Sigma": np.random.randn(50, 6, 6),
            "mean": np.random.randn(6, 50),
        }

        with patch("dcs.causality.rdcs.time_varying_causality") as mock_causality:
            mock_causality.return_value = {
                "TE": np.random.randn(50, 2),
                "DCS": np.random.randn(50, 2),
                "rDCS": np.random.randn(50, 2),
            }

            result = stage.execute(
                event_data=event_data,
                event_stats=event_stats,
                event_snapshots=event_data,  # Add the missing event_snapshots
                morder=4,
            )

            assert "causal_output" in result


class TestPipelineResult:
    """Test cases for PipelineResult class."""

    def test_pipeline_result_creation(self, minimal_config):
        """Test PipelineResult creation and attributes."""
        results = {"test_data": np.array([1, 2, 3])}
        event_snapshots = np.random.randn(2, 50, 10)

        pipeline_result = PipelineResult(
            results=results, config=minimal_config, event_snapshots=event_snapshots
        )

        assert pipeline_result.results == results
        assert pipeline_result.config == minimal_config
        assert np.array_equal(pipeline_result.event_snapshots, event_snapshots)

    def test_pipeline_result_to_dict(self, minimal_config):
        """Test PipelineResult to_dict method."""
        results = {"analysis_output": "test"}
        event_snapshots = np.random.randn(2, 30, 5)

        pipeline_result = PipelineResult(
            results=results, config=minimal_config, event_snapshots=event_snapshots
        )

        result_dict = pipeline_result.to_dict()
        assert "results" in result_dict
        assert "config" in result_dict
        assert "event_snapshots" in result_dict


class TestPipelineErrorHandling:
    """Test cases for pipeline error handling."""

    def test_computation_error_handling(self, minimal_config, sample_signals):
        """Test that computation errors are properly handled."""
        original_signal, detection_signal = sample_signals
        orchestrator = PipelineOrchestrator(minimal_config)

        # Mock a stage to raise an exception
        with patch.object(
            orchestrator.stages["input_validation"],
            "execute",
            side_effect=Exception("Test error"),
        ):
            with pytest.raises(ComputationError, match="Stage input_validation failed"):
                orchestrator.run(original_signal, detection_signal)

    def test_invalid_config_handling(self):
        """Test handling of invalid configuration."""
        # Should raise validation error during config validation
        with pytest.raises(ValueError, match="detection.locs must be provided"):
            PipelineConfig(
                options=PipelineOptions(detection=False),  # detection off
                detection=DetectionParams(
                    thres_ratio=1.0,
                    align_type="peak",
                    l_extract=30,
                    l_start=15,
                    locs=None,
                ),  # but no locations provided
                bic=BicParams(morder=4),
                causal=CausalParams(ref_time=25),
                output=OutputParams(file_keyword="test"),
            )


if __name__ == "__main__":
    pytest.main([__file__])
