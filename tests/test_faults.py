import numpy as np

from eegval.faults import (
    inject_clock_drift,
    inject_contact_loss,
    inject_emg_burst,
    inject_packet_dropout,
    inject_step_pop,
)
from eegval.metrics import missing_sample_percentage


def test_injectors_create_expected_fault_signatures():
    sfreq = 256
    signal = np.zeros(sfreq * 4)

    pop = inject_step_pop(signal, sfreq, 1, amplitude_uv=250)
    assert pop[sfreq] == 250e-6

    emg = inject_emg_burst(signal, sfreq, 1, 1, random_state=3)
    assert np.any(emg[sfreq : 2 * sfreq] != 0)

    dropout = inject_packet_dropout(signal, sfreq, 1, 0.5)
    assert missing_sample_percentage(dropout) == 12.5

    flatline = inject_contact_loss(np.arange(signal.size, dtype=float), sfreq, 1, 1)
    assert np.all(flatline[sfreq : 2 * sfreq] == flatline[sfreq])

    drifted = inject_clock_drift(np.sin(np.arange(signal.size) / 8), sfreq, 1000)
    assert drifted.shape == signal.shape