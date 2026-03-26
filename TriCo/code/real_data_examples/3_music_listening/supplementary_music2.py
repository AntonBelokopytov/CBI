# -*- coding: utf-8 -*-

import mne

# %%
# Load raw EDF data
dpath = 'D:/OS(CURRENT)/data/music/exp2/21.03_g1/MihKir_2224_1_aligned.fif'

# Load EDF file with data preloaded into memory
raw_init = mne.io.read_raw_fif(dpath, preload=True)

# Visual inspection of the raw data
raw_init.plot()

# %%
dpath_clean = 'D:/OS(CURRENT)/data/music/exp2/21.03_g1/MihKir_2224_1-clear.fif'
raw_clean = mne.io.read_raw_fif(dpath_clean, preload=True)

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
raw = mne.io.RawArray(data, info).resample(raw_clean.info['sfreq'])

# Attach annotations to the new Raw object
raw.set_annotations(annotations)

# %%
raw.plot()

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

raw.plot()
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
anns = ['LSL RS_E&#1057;', 'LSL RS_EO', 'LSL 2Hz',
       'LSL 05Hz', 'LSL 4Hz', 'LSL 1Hz', 'LSL 3Hz', 'LSL NoRy_1',
       'LSL Waltz_1', 'LSL Waltz_2', 'LSL NoRy_2', 'LSL NoRy_3',
       'LSL Waltz_3', 'LSL NoRy_4', 'LSL Waltz_4', 'LSL NoRy_5',
       'LSL Waltz_5', 'LSL RS_E&#1057;', 'LSL RS_EO', 'LSL Waltz_6',
       'LSL Waltz_7', 'LSL Waltz_8']

description = []
onset = []
duration = []
for ann in raw.annotations:
    if ann['description'] in anns:
        if ann['description'] == 'LSL RS_E&#1057;':
            description.append('RS_EC')
        else:
            description.append(ann['description'][4:])
        onset.append(ann['onset'])
        duration.append(120)

new_annotations = mne.Annotations(onset,duration,description,raw.annotations.orig_time)
new_annotations

# %%
raw_annotated = raw.copy().set_annotations(new_annotations)

# %%
segments = raw_annotated.crop_by_annotations()

# %%
raw_by_annotations = mne.concatenate_raws(segments)

# %%
raw_by_annotations.resample(raw_clean.info['sfreq'])

# %%
raw_by_annotations.plot()

# %%
picks_non_eeg = mne.pick_types(
    raw_by_annotations.info,
    eeg=False,
    ecg=True,
    misc=True,
    eog=True,
    gsr=True,
    resp=True,
)

raw_non_eeg = raw_by_annotations.copy().pick(picks_non_eeg)
raw_non_eeg.plot()

# %%
raw_clean.add_channels([raw_non_eeg], force_update_info=True)

# %%
raw_clean.plot()

# %%
raw_clean.save('D:/OS(CURRENT)/data/music/exp2/20.03_g1/Tumyalis_clear_with_peryph.fif')

# %%

