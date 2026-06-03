# -*- coding: utf-8 -*-
"""
Created on Tue Jun  2 18:31:10 2026

@author: ansbel
"""

import mne
import pandas as pd
import numpy as np

# %%
fpath = 'C:/Users/ansbel/Documents/GitHub/TriCo/data/external/music_listening/part1/eeg/'
fname = '10_07_g1_2223_raw.fif'

raw = mne.io.read_raw_fif(fpath + fname, preload=True)
fig = raw.plot(block=True,n_channels=38,duration=25)

# %%
raw.save(fpath + fname, overwrite=True)
