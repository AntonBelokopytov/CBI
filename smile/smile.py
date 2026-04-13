# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import mne

# %%
fpath = 'D:/OS(CURRENT)/data/smile_lobe/cleaned_data/E001_ptp150_ica_interpolated_raw.fif'

raw = mne.io.read_raw_fif(fpath,preload=True)

# %%
raw.plot()

# %%
