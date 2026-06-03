# -*- coding: utf-8 -*-
"""
Created on Tue May 26 23:06:39 2026

@author: anton
"""

import mne
import pandas as pd
import numpy as np

# %% [1] Загрузка данных
fname = '2224'
fpath = 'D:/Hyperscanning/Data/10.07/group 3/converted_to_fif/4. epochs_ica/NVX52_' + fname + '_ica_epochs.fif'
# fpath = 'D:/Hyperscanning/Data/9.07/group 1/NVX52_' + fname + '_ica_epochs.fif'
epochs = mne.read_epochs(fpath, preload=True)

target_conditions = [
    'EC1', 'EO1', '2Hz', '05Hz', '4Hz', '1Hz', '3Hz', 
    'NoRy 1', 'Waltz 1', 'Waltz 2', 'NoRy 2', 'NoRy 3', 
    'Waltz 3', 'NoRy 4', 'Waltz 4', 'NoRy 5', 'Waltz 5', 
    'EC2', 'EO2'
]

epochs.info

# %% [2] Проверка словаря и хронологии (Sanity Check)
events_array = epochs.events[:, 2]

print("ЗАПУСК ПРОВЕРКИ СООТВЕТСТВИЯ ТРИГГЕРОВ:")
print("-" * 90)
for i in range(len(events_array)):
    trigger = events_array[i]
    expected_name = target_conditions[i]
    
    # Ищем все текстовые ключи в event_id, соответствующие текущему триггеру
    names_in_dict = [key for key, val in epochs.event_id.items() if val == trigger]
    names_str = " ИЛИ ".join(names_in_dict)
    
    print(f"Индекс {i:2d} | Триггер в файле: {trigger:2d} | Ожидаем: {expected_name:7s} | В словаре MNE: {names_str}")
print("-" * 90)

# %% [2] Прямая конкатенация матриц ЭЭГ-сигнала в RawArray
# Извлекаем все эпохи по порядку и сшиваем по оси времени (axis=1)
data = epochs.get_data(copy=False)
raw_data = np.concatenate(data, axis=1)

# Создаем структуру Raw
info = epochs.info
raw = mne.io.RawArray(raw_data, info)

# %% [3] Генерация и наложение временных аннотаций
sfreq = info['sfreq']
epoch_duration = len(epochs.times) / sfreq             # Длительность одной эпохи в секундах

# Рассчитываем время начала для каждой эпохи (0s, 115s, 230s и т.д.)
onsets = [i * epoch_duration for i in range(len(target_conditions))]
durations = [epoch_duration] * len(target_conditions)

# Создаем объект чистых текстовых аннотаций
annotations = mne.Annotations(
    onset=onsets,
    duration=durations,
    description=target_conditions
)

raw.set_annotations(annotations)

# %%
raw.plot()

# %% [4] Визуализация и сохранение результатов
out_fpath = 'C:/Users/ansbel/Documents/GitHub/TriCo/data/external/music_listening/part1/' + fname + '_raw.fif'
raw.save(out_fpath, overwrite=True)



# %%
# %%
# %%
# %%
# %% [2] Умная сортировка и конкатенация матриц ЭЭГ-сигнала

# 1. Карта перевода: как твои целевые условия называются в словаре файла
name_map = {
    'EC1': 'Close your eyes1.mp3', 'EO1': 'Open your eyes1.mp3',
    '2Hz': '2Hz.mp3', '05Hz': '05Hz.mp3', '4Hz': '4Hz.mp3',
    '1Hz': '1Hz.mp3', '3Hz': '3Hz.mp3',
    'NoRy 1': 'NoRy 1.mp3', 'Waltz 1': 'Waltz 1.mp3', 'Waltz 2': 'Waltz 2.mp3',
    'NoRy 2': 'NoRy 2.mp3', 'NoRy 3': 'NoRy 3.mp3', 'Waltz 3': 'Waltz 3.mp3',
    'NoRy 4': 'NoRy 4.mp3', 'Waltz 4': 'Waltz 4.mp3', 'NoRy 5': 'NoRy 5.mp3',
    'Waltz 5': 'Waltz 5.mp3',
    'EC2': 'Close your eyes2.mp3', 'EO2': 'Open your eyes2.mp3'
}

events_array = epochs.events[:, 2]
ordered_indices = []

# 2. Вычисляем правильный порядок индексов
for cond in target_conditions:
    dict_name = name_map[cond]                       # Берем имя с .mp3
    trigger_code = epochs.event_id[dict_name]        # Узнаем его числовой триггер
    
    # Находим, под каким реальным индексом (0-18) этот триггер лежит в массиве
    actual_idx = np.where(events_array == trigger_code)[0][0]
    ordered_indices.append(actual_idx)

# 3. Извлекаем данные и выстраиваем их в правильном целевом порядке
data_ordered = epochs.get_data(copy=False)[ordered_indices]

# 4. Теперь безопасно сшиваем отсортированные матрицы по оси времени (axis=1)
raw_data = np.concatenate(data_ordered, axis=1)

# Создаем структуру Raw
info = epochs.info
raw = mne.io.RawArray(raw_data, info)