# EEG Signal Validation

`eegval` is a modality-agnostic validation harness for EEG signal-quality
metrics and controlled fault injection. Metrics operate on signal arrays and
do not branch on sensor placement or modality. Inputs use volts; RMS outputs
are reported in microvolts.

## Included

- Zero-phase FIR bandpass filtering and non-overlapping window RMS.
- Artifact-window percentage and missing-sample percentage.
- Stage-wise relative spectral power, alpha peak ratio, and windowed Pearson
  correlation.
- Block-shuffle permutation p-values and Benjamini-Hochberg correction.
- Injectors for step pops, band-limited EMG bursts, packet gaps, contact loss,
  and clock drift.
- EEGLAB loading through optional MNE, bipolar derivation, and explicit
  expected-sample validation to catch known truncated recordings.

The alpha metric currently uses alpha-band peak PSD divided by mean flanking
PSD. Treat it as a documented proxy, not an exact Tautan implementation, until
the paper's equation and preprocessing details are checked against the source.
The repository does not include the recordings and annotations needed to
reproduce the target paper's Fig. 8C.

## Small real-data sample

Two real PhysioNet EEGMMIDB baseline recordings are included at
[`data/physionet_eegmmidb`](data/physionet_eegmmidb/README.md): subject S001,
eyes open (`S001R01.edf`) and eyes closed (`S001R02.edf`). Each is a 64-channel,
160 Hz recording of about one minute. They are useful for loader and
alpha-analysis smoke tests. The data note includes source, license, and
citation details.

## Install and test

```powershell
py -m pip install -e ".[test]"
pytest
```

To enable EEGLAB loading, install the optional extra:

```powershell
py -m pip install -e ".[eeglab]"
```

MNE can read the included EDF baselines with `mne.io.read_raw_edf`.

## Minimal example

```python
import numpy as np
from eegval.metrics import artifact_percentage, bandpass_fir

sfreq = 128
signal_v = np.zeros(sfreq * 30)
filtered_v = bandpass_fir(signal_v, sfreq)
bad_percent = artifact_percentage(filtered_v, sfreq)
```

`artifact_percentage` counts complete 10-second windows whose RMS exceeds
100 µV; incomplete trailing windows are excluded. NaN-containing windows are
excluded from this metric and should be paired with `missing_sample_percentage`
so dropouts are not mistaken for clean data.

## Next validation steps

1. Add a metadata-aware loader workflow for representative recordings; check
   file integrity and expected sample counts before processing.
2. Reproduce the target paper's stage-wise correlation figure for documented
   channel pairs; record channel order, reference, filtering, windowing, and
   lag assumptions alongside the result.
3. Verify the alpha equation and permutation protocol against the paper, then
   add golden fixtures and benchmark fault detection by severity and false
   positive rate.
4. Add a CI job to run tests and regenerate deterministic figures after the
   real-data workflow and compact, redistributable fixtures are established.

The paper's reported abstract/results correlation percentages, the
Landis-Koch cutoff, and the RMS significance statement should be recorded as
replication discrepancies rather than silently reconciled.
