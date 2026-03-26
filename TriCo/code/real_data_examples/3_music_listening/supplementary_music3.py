# -*- coding: utf-8 -*-

import mne

# %%
# Load raw EDF data
dpath = 'D:/OS(CURRENT)/data/music/exp2/22.03_g1/DokNik_2299_2.edf'

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

info['subject_info'] = {
    'his_id': 'DokNik'
}

# %%
# Copy annotations from the original file
onsets = raw_init.annotations.onset
durations = raw_init.annotations.duration
descriptions = raw_init.annotations.description

annotations = mne.Annotations(onsets, durations, descriptions)

# %%
# Create a new Raw object with reconstructed metadata
raw = mne.io.RawArray(data, info).resample(500)
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
anns = ['LSL RS_EС1', 'LSL RS_EO1', 'LSL 2Hz',
       'LSL 05Hz', 'LSL 4Hz', 'LSL 1Hz', 'LSL 3Hz', 'LSL NoRy_1',
       'LSL Waltz_1', 'LSL Waltz_2', 'LSL NoRy_2', 'LSL NoRy_3',
       'LSL Waltz_3', 'LSL NoRy_4', 'LSL Waltz_4', 'LSL NoRy_5',
       'LSL Waltz_5', 'LSL RS_EС2', 'LSL RS_EO2', 'LSL Waltz_6',
       'LSL Waltz_7', 'LSL Waltz_8']

description = []
onset = []
duration = []
for ann in raw.annotations:
    print(ann['description'])
    if ann['description'] in anns:
        if ann['description'] == 'LSL RS_EС1' or ann['description'] == 'LSL RS_EС2':
            description.append('RS_EC')
        elif ann['description'] == 'LSL RS_EO1' or ann['description'] == 'LSL RS_EO2':
            description.append('RS_EO')
        else:
            description.append(ann['description'][4:])
        onset.append(ann['onset'])
        duration.append(120)

new_annotations = mne.Annotations(onset,duration,description,raw.annotations.orig_time)
new_annotations

raw_annotated = raw.copy().set_annotations(new_annotations)
print(raw_annotated.annotations.description)
print(new_annotations)

# %%
sfreq = raw.info['sfreq']
data = raw_annotated.get_data()

segments = []

for onset, duration, desc in zip(raw_annotated.annotations.onset,
                                  raw_annotated.annotations.duration,
                                  raw_annotated.annotations.description):

    start = int(onset * sfreq)
    stop  = int((onset + duration) * sfreq)

    seg_data = data[:, start:stop]

    seg = mne.io.RawArray(seg_data, raw.info.copy())

    seg_duration = (stop - start) / sfreq

    annot = mne.Annotations(
        onset=[0.0],
        duration=[seg_duration],
        description=[desc]
    )

    seg.set_annotations(annot)

    segments.append(seg)

raw_by_annotations = mne.concatenate_raws(segments)

# %%
raw_by_annotations.plot()

# %%
# индекс канала Sound
idx = raw_by_annotations.ch_names.index('Sound')
# убедиться, что длина совпадает
assert raw_by_annotations.n_times == music_chan.n_times
# заменить данные
mdata = music_chan.get_data()[0]
raw_by_annotations._data[idx, :mdata.shape[0]] = mdata

# %%
raw_by_annotations.plot()

# %%
fpath = 'D:/OS(CURRENT)/data/music/exp2/22.03_g1/DokNik_aligned.fif'
raw_by_annotations.save(fpath)

# %%
check_raw = mne.io.read_raw_fif(fpath,preload=True)
check_raw.plot()

# %%
music_chan = raw_by_annotations.copy().pick_channels(["Sound"])

# %%
music_chan.plot()

