"""Controlled signal-fault injectors."""

from eegval.faults.inject import (
    inject_clock_drift,
    inject_contact_loss,
    inject_emg_burst,
    inject_packet_dropout,
    inject_step_pop,
)

__all__ = [
    "inject_clock_drift",
    "inject_contact_loss",
    "inject_emg_burst",
    "inject_packet_dropout",
    "inject_step_pop",
]