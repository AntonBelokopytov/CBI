# -*- coding: utf-8 -*-
"""
Created on Tue May 26 23:06:39 2026

@author: anton
"""

import mne
import pandas as pd

# %%
fpath = 'D:/Hyperscanning/Data/10.07/group 2/converted_to_fif/4. ica_epochs/NVX52_2219_ica_epochs.fif'
epochs = mne.read_epochs(fpath,preload=True)

# %%
epochs.plot()

# %%
id_to_name = {v: k for k, v in epochs.event_id.items()}

# 2. Сопоставляем коды каждой эпохи по порядку с их названиями
ordered_epoch_names = [id_to_name[code] for code in epochs.events[:, 2]]
ordered_epoch_names

# %%

