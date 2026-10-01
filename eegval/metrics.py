"""Signal-quality metrics for one-dimensional EEG signals in volts."""

from collections.abc import Mapping

import numpy as np
from scipy.integrate import trapezoid
from scipy.signal import firwin, filtfilt, welch
from scipy.stats import pearsonr

DEFAULT_BANDS = {
    "delta": (0.5, 4.0),
    "theta": (4.0, 8.0),
    "alpha": (8.0, 13.0),
    "beta1": (13.0, 20.0),
    "beta2": (20.0, 35.0),
}


def bandpass_fir(
    signal_v: np.ndarray,
    sfreq: float,
    low_hz: float = 0.3,
    high_hz: float = 35.0,
    numtaps: int = 129,
) -> np.ndarray:
    """Apply a zero-phase FIR bandpass along the last axis."""
    data = np.asarray(signal_v, dtype=float)
    if data.ndim == 0 or sfreq <= 0 or not 0 < low_hz < high_hz < sfreq / 2:
        raise ValueError("Require data, positive sfreq, and 0 < low < high < Nyquist")
    if numtaps < 3 or numtaps % 2 == 0:
        raise ValueError("numtaps must be an odd integer of at least 3")
    if data.shape[-1] <= 3 * numtaps:
        raise ValueError("Signal is too short for the selected FIR filter")
    taps = firwin(numtaps, [low_hz, high_hz], pass_zero=False, fs=sfreq)
    return filtfilt(taps, [1.0], data, axis=-1)


def window_rms_uv(
    signal_v: np.ndarray,
    sfreq: float,
    window_seconds: float = 10.0,
    amplitude_scale: float = 1e6,
) -> np.ndarray:
    """Compute RMS for complete non-overlapping windows, in microvolts."""
    signal = np.asarray(signal_v, dtype=float)
    if signal.ndim != 1 or sfreq <= 0 or window_seconds <= 0:
        raise ValueError("Expected a 1-D signal and positive sampling/window rates")
    samples_per_window = int(round(sfreq * window_seconds))
    if samples_per_window < 1:
        raise ValueError("Window must contain at least one sample")
    count = signal.size // samples_per_window
    if count == 0:
        return np.empty(0, dtype=float)
    windows = signal[: count * samples_per_window].reshape(count, samples_per_window)
    valid = np.all(np.isfinite(windows), axis=1)
    rms = np.full(count, np.nan, dtype=float)
    rms[valid] = np.sqrt(np.mean(np.square(windows[valid]), axis=1)) * amplitude_scale
    return rms


def artifact_percentage(
    signal_v: np.ndarray,
    sfreq: float,
    threshold_uv: float = 100.0,
    window_seconds: float = 10.0,
) -> float:
    """Percent of valid complete windows with RMS above the amplitude limit."""
    rms = window_rms_uv(signal_v, sfreq, window_seconds)
    valid = np.isfinite(rms)
    if not np.any(valid):
        return float("nan")
    return float(np.mean(rms[valid] > threshold_uv) * 100.0)


def missing_sample_percentage(signal: np.ndarray) -> float:
    """Percent of samples that are non-finite, including packet-gap NaNs."""
    data = np.asarray(signal)
    if data.size == 0:
        return float("nan")
    return float(np.mean(~np.isfinite(data)) * 100.0)


def _band_power(frequencies: np.ndarray, psd: np.ndarray, band: tuple[float, float]):
    mask = (frequencies >= band[0]) & (frequencies < band[1])
    if np.count_nonzero(mask) < 2:
        return 0.0
    return float(trapezoid(psd[mask], frequencies[mask]))


def relative_power_by_stage(
    signal_v: np.ndarray,
    sfreq: float,
    stage_labels: np.ndarray,
    bands: Mapping[str, tuple[float, float]] = DEFAULT_BANDS,
) -> dict[object, dict[str, float]]:
    """Calculate percentage band power per stage from per-sample labels."""
    signal = np.asarray(signal_v, dtype=float)
    labels = np.asarray(stage_labels)
    if signal.ndim != 1 or labels.ndim != 1 or signal.size != labels.size:
        raise ValueError("Signal and per-sample stage labels must be matching 1-D arrays")
    if sfreq <= 0 or not bands:
        raise ValueError("Sampling rate and band definitions must be positive/nonempty")
    result = {}
    for stage in np.unique(labels):
        stage_data = signal[(labels == stage) & np.isfinite(signal)]
        if stage_data.size < 4:
            continue
        frequencies, psd = welch(stage_data, fs=sfreq, nperseg=min(256, stage_data.size))
        powers = {name: _band_power(frequencies, psd, band) for name, band in bands.items()}
        total = sum(powers.values())
        if total > 0:
            result[stage.item() if hasattr(stage, "item") else stage] = {
                name: power / total * 100.0 for name, power in powers.items()
            }
    return result


def alpha_peak_ratio(
    signal_v: np.ndarray,
    sfreq: float,
    alpha_band: tuple[float, float] = (8.0, 13.0),
    flank_bands: tuple[tuple[float, float], tuple[float, float]] = ((5.0, 7.5), (13.5, 20.0)),
    clear_peak_cutoff: float = 1.5,
) -> tuple[float, bool]:
    """Return alpha peak / mean flank PSD and whether it exceeds the cutoff.

    This is a transparent peak-prominence proxy, pending verification of the
    exact Tautan ratio definition used by the target paper.
    """
    signal = np.asarray(signal_v, dtype=float)
    if signal.ndim != 1 or not np.all(np.isfinite(signal)) or sfreq <= 0:
        raise ValueError("Expected a finite 1-D signal and positive sampling rate")
    frequencies, psd = welch(signal, fs=sfreq, nperseg=min(1024, signal.size))
    alpha_mask = (frequencies >= alpha_band[0]) & (frequencies <= alpha_band[1])
    flank_mask = np.zeros(frequencies.shape, dtype=bool)
    for low, high in flank_bands:
        flank_mask |= (frequencies >= low) & (frequencies <= high)
    if not np.any(alpha_mask) or not np.any(flank_mask):
        raise ValueError("Signal frequency resolution does not cover alpha and flank bands")
    flank_mean = float(np.mean(psd[flank_mask]))
    ratio = float(np.max(psd[alpha_mask]) / flank_mean) if flank_mean > 0 else float("inf")
    return ratio, ratio >= clear_peak_cutoff


def windowed_correlation(
    first: np.ndarray,
    second: np.ndarray,
    sfreq: float,
    window_seconds: float = 10.0,
) -> np.ndarray:
    """Return Pearson correlation for complete, aligned non-overlapping windows."""
    first = np.asarray(first, dtype=float)
    second = np.asarray(second, dtype=float)
    if first.ndim != 1 or second.ndim != 1 or first.size != second.size:
        raise ValueError("Signals must be matching 1-D arrays")
    samples_per_window = int(round(sfreq * window_seconds))
    if samples_per_window < 2:
        raise ValueError("Each window must contain at least two samples")
    count = first.size // samples_per_window
    correlations = np.full(count, np.nan)
    for index in range(count):
        start = index * samples_per_window
        stop = start + samples_per_window
        x = first[start:stop]
        y = second[start:stop]
        if np.all(np.isfinite(x)) and np.all(np.isfinite(y)):
            correlations[index] = pearsonr(x, y).statistic
    return correlations


def block_shuffle_pvalue(
    first: np.ndarray,
    second: np.ndarray,
    block_samples: int,
    n_surrogates: int = 1600,
    random_state: int | np.random.Generator | None = None,
) -> float:
    """Two-sided Pearson permutation p-value using shuffled blocks of signal 2."""
    first = np.asarray(first, dtype=float)
    second = np.asarray(second, dtype=float)
    if first.ndim != 1 or first.size != second.size or not np.all(np.isfinite(first + second)):
        raise ValueError("Inputs must be matching, finite 1-D signals")
    if block_samples < 1 or n_surrogates < 1:
        raise ValueError("Block size and surrogate count must be positive")
    blocks = [second[start : start + block_samples] for start in range(0, second.size, block_samples)]
    if len(blocks) < 2:
        raise ValueError("At least two blocks are required for the shuffle test")
    observed = abs(float(pearsonr(first, second).statistic))
    rng = random_state if isinstance(random_state, np.random.Generator) else np.random.default_rng(random_state)
    exceedances = 0
    for _ in range(n_surrogates):
        shuffled = np.concatenate([blocks[index] for index in rng.permutation(len(blocks))])
        surrogate = abs(float(pearsonr(first, shuffled).statistic))
        exceedances += surrogate >= observed
    return (exceedances + 1) / (n_surrogates + 1)


def fdr_bh(p_values: np.ndarray, alpha: float = 0.05) -> np.ndarray:
    """Return Benjamini-Hochberg rejection decisions, preserving input shape."""
    values = np.asarray(p_values, dtype=float)
    if not 0 < alpha < 1:
        raise ValueError("alpha must be between zero and one")
    if np.any((values[np.isfinite(values)] < 0) | (values[np.isfinite(values)] > 1)):
        raise ValueError("Finite p-values must be between zero and one")
    flat = values.ravel()
    valid_indices = np.flatnonzero(np.isfinite(flat))
    rejected = np.zeros(flat.shape, dtype=bool)
    if valid_indices.size:
        sorted_indices = valid_indices[np.argsort(flat[valid_indices])]
        sorted_values = flat[sorted_indices]
        bounds = alpha * np.arange(1, sorted_values.size + 1) / sorted_values.size
        passing = np.flatnonzero(sorted_values <= bounds)
        if passing.size:
            rejected[sorted_indices[: passing[-1] + 1]] = True
    return rejected.reshape(values.shape)