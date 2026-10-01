# PhysioNet EEGMMIDB Sample

This folder contains two one-minute baseline EEG recordings for subject S001:

- `S001R01.edf`: eyes-open baseline.
- `S001R02.edf`: eyes-closed baseline.

The files are from the PhysioNet EEG Motor Movement/Imagery Dataset,
version 1.0.0. They contain 64 EEG channels sampled at 160 Hz and are useful
for exercising the loader and eyes-open/closed alpha analysis. The analysis
package treats each channel as a named signal and does not depend on sensor
placement.

Source: <https://physionet.org/content/eegmmidb/1.0.0/>

The dataset is distributed under the Open Data Commons Attribution License
v1.0. Cite the dataset as:

Schalk, G. (2009). EEG Motor Movement/Imagery Dataset (version 1.0.0).
PhysioNet. https://doi.org/10.13026/C28G6P

The full dataset is much larger; only these two baseline files are included.