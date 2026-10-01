"""Tools for validating EEG signal quality without sensor-type assumptions."""

from eegval.metrics import (
    DEFAULT_BANDS,
    alpha_peak_ratio,
    artifact_percentage,
    bandpass_fir,
    fdr_bh,
    missing_sample_percentage,
    relative_power_by_stage,
    window_rms_uv,
    windowed_correlation,
)

__all__ = [
    "DEFAULT_BANDS",
    "alpha_peak_ratio",
    "artifact_percentage",
    "bandpass_fir",
    "fdr_bh",
    "missing_sample_percentage",
    "relative_power_by_stage",
    "window_rms_uv",
    "windowed_correlation",
]