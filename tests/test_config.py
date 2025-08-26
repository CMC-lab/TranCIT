import pytest

from trancit.config import (
    BicParams,
    CausalParams,
    DetectionParams,
    OutputParams,
    PipelineConfig,
    PipelineOptions,
)


def test_valid_config():
    config = PipelineConfig(
        options=PipelineOptions(detection=True),
        detection=DetectionParams(
            thres_ratio=1.0, align_type="peak", l_extract=10, l_start=5
        ),
        bic=BicParams(morder=1),
        causal=CausalParams(ref_time=0),
        output=OutputParams(file_keyword="test"),
    )
    assert isinstance(config, PipelineConfig)


def test_missing_locs_with_detection_off():
    with pytest.raises(
        ValueError,
        match="detection.locs must be provided if options.detection is False",
    ):
        PipelineConfig(
            options=PipelineOptions(detection=False),
            detection=DetectionParams(
                thres_ratio=1.0, align_type="peak", l_extract=10, l_start=5, locs=None
            ),
            bic=BicParams(morder=1),
            causal=CausalParams(ref_time=0),
            output=OutputParams(file_keyword="test"),
        )


def test_missing_bic_params():
    with pytest.raises(
        ValueError,
        match="bic.momax, bic.tau, and bic.mode must be set if options.bic is True",
    ):
        PipelineConfig(
            options=PipelineOptions(detection=True, bic=True),
            detection=DetectionParams(
                thres_ratio=1.0, align_type="peak", l_extract=10, l_start=5
            ),
            bic=BicParams(morder=1),
            causal=CausalParams(ref_time=0),
            output=OutputParams(file_keyword="test"),
        )


def test_missing_monte_carlo_with_bootstrap():
    with pytest.raises(
        ValueError,
        match="monte_carlo parameters must be provided if options.bootstrap is True",
    ):
        PipelineConfig(
            options=PipelineOptions(detection=True, bootstrap=True),
            detection=DetectionParams(
                thres_ratio=1.0, align_type="peak", l_extract=10, l_start=5
            ),
            bic=BicParams(morder=1),
            causal=CausalParams(ref_time=0),
            output=OutputParams(file_keyword="test"),
        )


def test_invalid_align_type():
    with pytest.raises(
        ValueError,
        match="Invalid detection.align_type: invalid. Must be 'peak' or 'pooled'.",
    ):
        PipelineConfig(
            options=PipelineOptions(detection=True),
            detection=DetectionParams(
                thres_ratio=1.0, align_type="invalid", l_extract=10, l_start=5
            ),
            bic=BicParams(morder=1),
            causal=CausalParams(ref_time=0),
            output=OutputParams(file_keyword="test"),
        )
