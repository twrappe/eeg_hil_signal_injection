import numpy as np
from scipy.stats import kstest

from eegval.io import TruncatedRecordingError, validate_sample_count
from eegval.metrics import (
    alpha_peak_ratio,
    artifact_percentage,
    block_shuffle_pvalue,
    fdr_bh,
    missing_sample_percentage,
    relative_power_by_stage,
    window_rms_uv,
    windowed_correlation,
)


def test_rms_and_artifact_percentage_use_complete_windows():
    signal = np.concatenate([np.full(100, 50e-6), np.full(100, 150e-6), np.ones(10)])
    np.testing.assert_allclose(window_rms_uv(signal, 10, window_seconds=10), [50, 150])
    assert artifact_percentage(signal, 10, window_seconds=10) == 50


def test_self_correlation_is_one():
    signal = np.random.default_rng(4).normal(size=400)
    np.testing.assert_allclose(windowed_correlation(signal, signal, 20, 10), [1, 1])


def test_block_shuffle_pvalue_is_reproducible_and_bounded():
    rng = np.random.default_rng(7)
    first = rng.normal(size=800)
    second = rng.normal(size=800)
    first_p = block_shuffle_pvalue(first, second, 40, 99, random_state=2)
    second_p = block_shuffle_pvalue(first, second, 40, 99, random_state=2)
    assert first_p == second_p
    assert 0 < first_p <= 1


def test_independent_noise_permutation_pvalues_are_approximately_uniform():
    rng = np.random.default_rng(17)
    p_values = [
        block_shuffle_pvalue(
            rng.normal(size=120), rng.normal(size=120), 10, 39, random_state=index
        )
        for index in range(40)
    ]
    assert kstest(p_values, "uniform").pvalue > 0.01


def test_stage_power_and_alpha_peak_metrics():
    sfreq = 128
    times = np.arange(sfreq * 20) / sfreq
    signal = 20e-6 * np.sin(2 * np.pi * 10 * times)
    stages = np.repeat([1, 2], signal.size // 2)
    power = relative_power_by_stage(signal, sfreq, stages)
    assert set(power) == {1, 2}
    assert power[1]["alpha"] > power[1]["delta"]
    ratio, clear_peak = alpha_peak_ratio(signal, sfreq)
    assert ratio > 1.5
    assert clear_peak


def test_fdr_bh_preserves_shape_and_ignores_nan():
    rejected = fdr_bh(np.array([[0.001, 0.02], [0.7, np.nan]]))
    np.testing.assert_array_equal(rejected, [[True, True], [False, False]])


def test_dropout_is_reported_as_missing():
    signal = np.array([0.0, np.nan, 1.0, np.nan])
    assert missing_sample_percentage(signal) == 50


def test_truncated_recording_is_not_silently_accepted():
    try:
        validate_sample_count(90, 100)
    except TruncatedRecordingError as error:
        assert "truncated" in str(error)
    else:
        raise AssertionError("Expected a truncated recording to fail validation")