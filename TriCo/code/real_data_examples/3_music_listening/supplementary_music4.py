# -*- coding: utf-8 -*-
"""
Created on Sat Feb 28 05:45:32 2026

@author: ansbel
"""

import mne

# %%
# Load raw EDF data
dpath1 = 'D:/OS(CURRENT)/data/music/exp2/22.03_g1/DokNik_2299_2.edf'

# Load EDF file with data preloaded into memory
raw_init = mne.io.read_raw_edf(dpath1)
raw_init.info

# %%
dpath2 = 'D:/OS(CURRENT)/data/music/exp2/22.03_g1/DokNik_aligned.fif'
raw_al = mne.io.read_raw_fif(dpath2,preload=True)
raw_al.info
raw_al.set_meas_date(raw_init.info['meas_date'])
raw_al.info

# %%
raw_al.info['subject_info'] = {
    'his_id': 'DokNik'
    }
raw_al.info

# %%
raw_al.save(dpath2, overwrite=True)

# %%


