# -*- coding: utf-8 -*-
"""
Created on Mon May 18 16:59:20 2026

@author: anton
"""

import os
import mne

# %% Загрузка данных
fpath = 'D:/OS(CURRENT)/data/music/exp2/22.03_g1/DokNik_2299_2-clear.fif'
raw = mne.io.read_raw_fif(fpath, preload=True)

# %% Целевые условия
folders = ['RS_EC1', 'RS_EO1', '2Hz', '05Hz', '4Hz', '1Hz',
           '3Hz', 'NoRy_1', 'Waltz_1', 'Waltz_2', 'NoRy_2', 'NoRy_3',
           'Waltz_3', 'NoRy_4', 'Waltz_4', 'NoRy_5', 'Waltz_5', 'RS_EC2', 
           'RS_EO2', 'Waltz_6', 'Waltz_7', 'Waltz_8']

# === ИЗМЕНЕНИЕ ===
# Получаем имя файла с расширением (например, 'DmiAna_2200_3-clear.fif')
filename_with_ext = os.path.basename(fpath)

# Берем часть до первого нижнего подчеркивания ('DmiAna') и добавляем '_edf'
subject_prefix = filename_with_ext.split('_')[0]
main_folder_name = f"{subject_prefix}_edf"

# Базовая папка для экспорта
base_dir = os.path.dirname(fpath)
base_out_path = os.path.join(base_dir, main_folder_name)
# =================

# %% Нарезка по 120 секунд и сохранение
chunk_duration = 120.0

for i, condition in enumerate(folders):
    # Вычисляем время начала и конца текущего отрезка
    tmin = i * chunk_duration
    tmax = (i + 1) * chunk_duration
    
    # Защита от выхода за пределы длительности записи
    if tmin >= raw.times[-1]:
        print(f"Достигнут конец файла! Пропуск условия: {condition}")
        break
        
    if tmax > raw.times[-1]:
        tmax = raw.times[-1]
        
    # Создаем копию и обрезаем ее
    raw_cropped = raw.copy().crop(tmin=tmin, tmax=tmax)
    
    # Формируем структуру директорий (используем i+1, чтобы папки начинались с 1)
    folder_name = f"{i + 1}_{condition}"
    folder_path = os.path.join(base_out_path, folder_name)
    os.makedirs(folder_path, exist_ok=True)
    
    # Формируем путь к файлу
    file_name = f"{condition}.edf"
    file_path = os.path.join(folder_path, file_name)
    
    # Экспортируем в EDF
    mne.export.export_raw(file_path, raw_cropped, fmt='edf', overwrite=True)
    
    print(f"[{tmin}с - {tmax}с] сохранено в: {file_path}")