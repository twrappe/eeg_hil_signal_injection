"""Fault injectors; time and amplitude inputs use seconds and volts."""

import numpy as np
from scipy.signal import butter, sosfiltfilt


def _sample_interval(size: int, sfreq: float, start_s: float, duration_s: float):
    if sfreq <= 0 or start_s < 0 or duration_s <= 0:
        raise ValueError("Require positive sampling rate/duration and nonnegative start")
    start = int(round(start_s * sfreq))
    stop = min(size, int(round((start_s + duration_s) * sfreq)))
    if start >= size or stop <= start:
        raise ValueError("Fault interval does not overlap the signal")
    return start, stop


def inject_step_pop(
    signal_v: np.ndarray, sfreq: float, start_s: float, amplitude_uv: float = 300.0
) -> np.ndarray:
    """Add a persistent electrode offset beginning at ``start_s``."""
    signal = np.asarray(signal_v, dtype=float).copy()
    if signal.ndim != 1 or sfreq <= 0:
        raise ValueError("Expected a 1-D signal and positive sampling rate")
    start, _ = _sample_interval(signal.size, sfreq, start_s, 1 / sfreq)
    signal[start:] += amplitude_uv * 1e-6
    return signal


def inject_emg_burst(
    signal_v: np.ndarray,
    sfreq: float,
    start_s: float,
    duration_s: float,
    amplitude_uv: float = 50.0,
    random_state: int | None = None,
) -> np.ndarray:
    """Add a seeded band-limited 20-100 Hz burst (upper band respects Nyquist)."""
    signal = np.asarray(signal_v, dtype=float).copy()
    if signal.ndim != 1 or sfreq <= 42:
        raise ValueError("Expected a 1-D signal sampled above 42 Hz")
    start, stop = _sample_interval(signal.size, sfreq, start_s, duration_s)
    upper_hz = min(100.0, sfreq / 2 - 1.0)
    sos = butter(4, [20.0, upper_hz], btype="bandpass", fs=sfreq, output="sos")
    rng = np.random.default_rng(random_state)
    noise = sosfiltfilt(sos, rng.standard_normal(stop - start))
    standard_deviation = float(np.std(noise))
    if standard_deviation > 0:
        noise = noise / standard_deviation * amplitude_uv * 1e-6
    signal[start:stop] += noise
    return signal


def inject_packet_dropout(
    signal_v: np.ndarray, sfreq: float, start_s: float, duration_s: float
) -> np.ndarray:
    """Represent missing packet samples explicitly as NaNs."""
    signal = np.asarray(signal_v, dtype=float).copy()
    if signal.ndim != 1:
        raise ValueError("Expected a 1-D signal")
    start, stop = _sample_interval(signal.size, sfreq, start_s, duration_s)
    signal[start:stop] = np.nan
    return signal


def inject_contact_loss(
    signal_v: np.ndarray,
    sfreq: float,
    start_s: float,
    duration_s: float,
    mode: str = "flatline",
    drift_uv: float = 200.0,
) -> np.ndarray:
    """Replace a segment with a flatline or linear rail-like drift."""
    signal = np.asarray(signal_v, dtype=float).copy()
    if signal.ndim != 1:
        raise ValueError("Expected a 1-D signal")
    start, stop = _sample_interval(signal.size, sfreq, start_s, duration_s)
    if mode == "flatline":
        signal[start:stop] = signal[start]
    elif mode == "drift":
        signal[start:stop] = np.linspace(
            signal[start], signal[start] + drift_uv * 1e-6, stop - start
        )
    else:
        raise ValueError("mode must be 'flatline' or 'drift'")
    return signal


def inject_clock_drift(
    signal_v: np.ndarray, sfreq: float, drift_ppm: float
) -> np.ndarray:
    """Resample a signal onto a clock drifting by ``drift_ppm`` parts per million."""
    signal = np.asarray(signal_v, dtype=float)
    if signal.ndim != 1 or sfreq <= 0 or abs(drift_ppm) >= 1_000_000:
        raise ValueError("Expected 1-D data, positive sampling rate, and valid drift")
    sample_times = np.arange(signal.size, dtype=float) / sfreq
    drifting_times = sample_times * (1 + drift_ppm * 1e-6)
    return np.interp(sample_times, drifting_times, signal, left=signal[0], right=signal[-1])