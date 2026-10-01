"""Optional EEGLAB loading and checks for incomplete recordings."""

from pathlib import Path
from typing import Mapping

import numpy as np


class TruncatedRecordingError(ValueError):
    """Raised when a recording contains fewer samples than its metadata."""


def validate_sample_count(actual_samples: int, expected_samples: int) -> None:
    """Fail explicitly when the loaded recording is shorter than expected."""
    if actual_samples < expected_samples:
        raise TruncatedRecordingError(
            f"Recording has {actual_samples} samples; expected at least "
            f"{expected_samples}. The source recording may be truncated."
        )


def derive_bipolar(
    channels: Mapping[str, np.ndarray], positive: str, negative: str
) -> np.ndarray:
    """Return a positive-minus-negative derivation from named channel arrays."""
    missing = [name for name in (positive, negative) if name not in channels]
    if missing:
        raise KeyError(f"Missing channel(s): {', '.join(missing)}")
    positive_data = np.asarray(channels[positive], dtype=float)
    negative_data = np.asarray(channels[negative], dtype=float)
    if positive_data.shape != negative_data.shape:
        raise ValueError("Bipolar channels must have matching shapes")
    return positive_data - negative_data


def load_eeglab_recording(
    set_file: str | Path,
    expected_samples: int | None = None,
    preload: bool = False,
):
    """Load an EEGLAB file with MNE and optionally validate its sample count.

    MNE is an optional dependency. For files with external ``.fdt`` data, MNE
    reports source-level read errors; this function additionally checks sample
    count against metadata supplied by the caller.
    """
    try:
        import mne
    except ImportError as error:
        raise ImportError(
            "EEGLAB loading requires MNE; install eegval with the 'eeglab' extra"
        ) from error

    raw = mne.io.read_raw_eeglab(str(set_file), preload=preload, verbose="ERROR")
    if expected_samples is not None:
        validate_sample_count(raw.n_times, expected_samples)
    return raw