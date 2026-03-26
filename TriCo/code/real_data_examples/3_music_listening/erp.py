# -*- coding: utf-8 -*-
"""
Example EEG processing pipeline using MNE-Python.

The script demonstrates:
- loading EDF data
- reconstructing Raw object and metadata
- setting montage and sensor positions
- filtering
- event detection from an auxiliary channel
- epoching
- ICA-based artifact correction
- ERP and topographic visualization
"""

import mne
import matplotlib.pyplot as plt
import numpy as np

# %%
# Load raw EDF data

dpath = 'D:/OS(CURRENT)/data/music/exp2/20.03_g1/Tumyalis_clear.fif'

# Load EDF file with data preloaded into memory
raw_init = mne.io.read_raw_fif(dpath, preload=True)

# Visual inspection of the raw data
raw_init.plot_sensors()

# %%
import os
import re

dpath = r'D:/OS(CURRENT)/data/music/exp2/20.03_g1/music_listening/'

for idx, ann in enumerate(raw_init.annotations, start=1):
    ons = ann['onset']
    descr = ann['description']

    # очистка имени от недопустимых символов
    safe_descr = re.sub(r'[^\w\-]', '_', descr)

    # нумерация с ведущими нулями
    folder_name = f'{idx}_{safe_descr}'
    folder_path = os.path.join(dpath, folder_name)

    os.makedirs(folder_path, exist_ok=True)

    raw_edf = raw_init.copy().crop(tmin=ons, tmax=ons + 120)

    file_path = os.path.join(folder_path, f'{safe_descr}.edf')
    raw_edf.export(file_path, fmt='edf')
    
# %%
dpath = 'D:/OS(CURRENT)/data/music/exp2/20.03_g1/music_listening/3_2Hz/2Hz.edf'
raw_init = mne.io.read_raw_edf(dpath, preload=True)
raw_init.plot()

# %%
raw_init.plot_sensors()


# %%
# Crop data to a time interval of interest
# Crop 120 seconds starting from t = 610 s
raw_cr1 = raw_init.copy().crop(tmin=360, tmax=480)
raw_cr2 = raw_init.copy().crop(tmin=600, tmax=720)
raw_cr = mne.concatenate_raws([raw_cr1,raw_cr2])

# %%
# Temporal filtering
# Notch filter to suppress line noise
rawf = raw_cr.copy().notch_filter([50, 100])

# Band-pass filter
rawf.filter(l_freq=0.1, h_freq=100)

# Power spectral density for quality control
rawf.compute_psd(fmin=1, fmax=25).plot()

# %%
idx = rawf.ch_names.index('Sound')
sound_data = rawf.get_data()[idx, :]

# %%
# Plot auxiliary signal
plt.plot(sound_data)

# %%
sfreq = rawf.info['sfreq']
# Detect threshold crossings
ts_above = np.where(sound_data > 0.1)[0]

# Keep only isolated events (minimum 1 second apart)
timestamps = [ts_above[0]]
for ts in ts_above[1:]:
    if ts - timestamps[-1] > rawf.info['sfreq'] * 0.5:
        timestamps.append(ts)

timestamps = np.array(timestamps)

# Create annotations for detected events
new_anns = mne.Annotations(timestamps / sfreq, [0], 'beep')

# Attach annotations to data
rawf.set_annotations(new_anns)

rawf.plot()

# %%
# Epoching

# Create MNE-style events array
events = np.stack(
    [timestamps + rawf.first_samp,
     np.zeros(len(timestamps)),
     np.ones(len(timestamps))],
    axis=1
).astype(int)

event_id = {'beep': 1}

# Extract epochs around detected events
epochs = mne.Epochs(
    rawf,
    events,
    event_id,
    tmin=-0.2,
    tmax=1,
    baseline=(None, 0),
    preload=True
)


# # %%
# # ICA for artifact correction
# # Initialize ICA
# ica = ICA(
#     n_components=38,
#     method='fastica'
# )

# # Fit ICA on epoched data
# ica.fit(epochs)

# # Visualize ICA source activations
# ica.plot_sources(rawf)

# # %%
# # Apply ICA to the epoched data
# epochsf = epochs.copy()
# ica.apply(epochsf)

# epochsf.plot()

# %%
epochs.save('erp.fif')

# %%
# ERP computation
# Compute averaged ERP
erp = epochs['beep'].average()

# Plot ERP waveforms
erp.plot()


# %%
# ERP scalp topographies

# Plot topographic maps at selected latencies
erp.plot_topomap(times=[0, 0.1, 0.2], average=0.05)


# %%
# Current Source Density (CSD) / surface Laplacian
from mne.preprocessing import compute_current_source_density

# Apply CSD transform to ERP
erp_csd = compute_current_source_density(erp)

# Plot CSD topographies
erp_csd.plot_topomap(times=[0.1, 0.2])

  
