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
from datetime import datetime
import matplotlib.pyplot as plt
import numpy as np
from mne.preprocessing import ICA

# %%
# Load raw EDF data

dpath = 'D:/OS(CURRENT)/data/music/exp2/21.03_g2/BerMih_2337_1.edf'

# Load EDF file with data preloaded into memory
raw_init = mne.io.read_raw_edf(dpath, preload=True)

# Visual inspection of the raw data
raw_init.plot()

# %%
# Extract data and basic metadata

# Numerical data array (n_channels × n_samples)
data = raw_init.get_data()

# Channel names
ch_names = raw_init.ch_names

# Sampling frequency
sfreq = raw_init.info['sfreq']

# Channel types:
# EEG channels first, auxiliary channels afterwards
ch_types = ['eeg'] * 38 + ['ecg'] + ['misc'] + ['gsr'] + ['resp'] + ['misc']

# %%
# Create MNE Info object
info = mne.create_info(ch_names, sfreq, ch_types)

# %%
# Copy annotations from the original file
onsets = raw_init.annotations.onset
durations = raw_init.annotations.duration
descriptions = raw_init.annotations.description

annotations = mne.Annotations(onsets, durations, descriptions)

# %%
# Create a new Raw object with reconstructed metadata
raw = mne.io.RawArray(data, info)

# Attach annotations to the new Raw object
raw.set_annotations(annotations)

# %%
# raw.plot()

# %%
# Set EEG montage (electrode positions)
# Load standard 10–20 montage
montage = mne.channels.make_standard_montage('standard_1020')

# Print channel names available in this montage
print(montage.ch_names)

# Apply montage to data
raw.set_montage(montage)

# %%
# Rename channels to match montage naming convention
ch_rename = {
    'Ft7': 'FT7',
    'Fc3': 'FC3',
    'Fcz': 'FCz',
    'Fc4': 'FC4',
    'Ft8': 'FT8',
    'Tp7': 'TP7',
    'Cp3': 'CP3',
    'Cpz': 'CPz',
    'Cp4': 'CP4',
    'Tp8': 'TP8',
    'Po3': 'PO3',
    'Poz': 'POz',
    'Po4': 'PO4',
    'Po7': 'PO7',
    'Po8': 'PO8'
}

raw.rename_channels(ch_rename)

# Re-apply montage after renaming
raw.set_montage(montage)

# %%
# Visual inspection of signals and sensor positions

# raw.plot()
raw.plot_sensors()

# %%
# Manual adjustment of sensor positions (demonstration only)

# Extract current montage
montage = raw.get_montage()
pos = montage.get_positions()['ch_pos']

# Shift all electrodes 2 cm posteriorly along the Y axis
# (demonstrational example, not physiologically motivated)
for ch in pos:
    pos[ch][1] -= 0.02  # shift by -20 mm

# Create a new montage with modified positions
new_montage = mne.channels.make_dig_montage(
    ch_pos=pos, coord_frame='head'
)

# Apply modified montage
raw.set_montage(new_montage)

# Plot updated sensor positions
raw.plot_sensors(show_names=True)

# %%
raw.compute_psd(fmin=1,fmax=100).plot()

# %%
# Crop data to a time interval of interest
# Crop 120 seconds starting from t = 610 s
raw_cr = raw.copy().crop(tmin=670, tmax=670 + 120)

# %%
# Temporal filtering
# Notch filter to suppress line noise
rawf = raw_cr.copy().notch_filter([50, 100])

# Band-pass filter
rawf.filter(l_freq=0.1, h_freq=25)

# Power spectral density for quality control
rawf.compute_psd(fmin=1, fmax=50).plot()

# %%
# Event detection from an auxiliary channel

# Extract the last channel (e.g., sound or trigger channel)
sound_data = rawf.get_data()[-1, :]

# %%
# Plot auxiliary signal
plt.plot(sound_data)

# %%
# Detect threshold crossings
ts_above = np.where(sound_data > 0.1)[0]

# Keep only isolated events (minimum 1 second apart)
timestamps = [ts_above[0]]
for ts in ts_above[1:]:
    if ts - timestamps[-1] > rawf.info['sfreq'] * 1:
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
    tmin=-1,
    tmax=1,
    baseline=(None, 0),
    preload=True
)

# epochs.plot()

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

  
# %% Simple statistics: compare ERP against zero
import scipy.stats as stats

# Define the time window of interest (in seconds)
tmin, tmax = 0.08, 0.12

# Get indices for this time window
times = epochsf.times
time_inds = np.where((times >= tmin) & (times <= tmax))[0]

# Get all epochs data (n_epochs × n_channels × n_times)
data_ep = epochsf.get_data()  # shape: (n_epochs, n_channels, n_times)

# Average over time within the window
data_mean = data_ep[:, :, time_inds].mean(axis=2)  # shape: (n_epochs, n_channels)

# One-sample t-test against zero for each channel
t_vals, p_vals = stats.ttest_1samp(data_mean, 0, axis=0)

# Print significant channels
alpha = 0.05
sig_chs = np.array(epochsf.ch_names)[p_vals < alpha]
print(f"Significant channels (p<{alpha}):", sig_chs)

# Compute mean ERP across epochs
erp_mean = data_mean.mean(axis=0)

# Create an Evoked object for plotting
evoked_array = mne.EvokedArray(erp_mean[:, np.newaxis],
                               epochsf.info, tmin=0)

# Create a mask for significant channels
mask = p_vals < alpha

# Plot topography at the center of the window, marking significant channels with asterisks
evoked_array.plot_topomap(times=0, mask=mask[:, np.newaxis],
                          mask_params=dict(marker='*', markersize=12, markerfacecolor='red'),
                           )

# %%


 