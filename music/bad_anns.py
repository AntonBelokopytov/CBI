# -*- coding: utf-8 -*-
"""
Скрипт для извлечения аннотаций BADS из MNE .fif файлов 
и их экспорта в MATLAB (.mat)
"""

import mne
import os
import numpy as np
import glob
from scipy.io import savemat

# 1. Задаем пути
fpath = 'C:/Users/ansbel/Documents/GitHub/TriCo/data/external/music_listening/part2/'

# Ищем все файлы _raw.fif в папке (основываясь на твоем скриншоте)
file_pattern = os.path.join(fpath, '*_raw.fif')
fif_files = glob.glob(file_pattern)

# Словарь, который превратится в структуру MATLAB
# Ключ - имя файла/испытуемого, Значение - массив Nx2
matlab_dict = {}

# 2. Проходимся по каждому файлу
for file in fif_files:
    # Достаем базовое имя файла (например, '26_07_g1_2224')
    fname = os.path.basename(file)
    base_name = fname.replace('_raw.fif', '')
    
    # MATLAB не позволяет называть переменные с цифры, поэтому добавляем префикс 's_'
    # Итог: 's_26_07_g1_2224'
    mat_field_name = 's_' + base_name 
    
    print(f"Обработка файла: {fname} -> ключ MATLAB: {mat_field_name}")
    
    # Загружаем файл (preload=False, так как нам не нужны сами данные ЭЭГ, только метаданные)
    raw = mne.io.read_raw_fif(file, preload=False, verbose='ERROR')
    
    bads_onsets = []
    bads_durations = []
    
    # 3. Перебираем все аннотации и ищем плохие участки
    for annot in raw.annotations:
        desc = annot['description']
        # Проверяем, содержит ли описание метку BAD (в MNE по умолчанию 'BAD_', у тебя 'BADS_')
        if 'BAD' in desc.upper():
            bads_onsets.append(annot['onset'])
            bads_durations.append(annot['duration'])
            
    # 4. Формируем массив Nx2
    if len(bads_onsets) > 0:
        # Объединяем списки в двумерный массив NumPy (Nx2)
        bads_array = np.column_stack((bads_onsets, bads_durations))
    else:
        # Если артефактов нет (как у Tumyalis: []), создаем пустой массив
        bads_array = np.array([]) 
        
    # Сохраняем массив в словарь
    matlab_dict[mat_field_name] = bads_array

# 5. Экспортируем весь словарь в MATLAB файл
out_mat_file = os.path.join(fpath, 'BADS_all_files.mat')
savemat(out_mat_file, matlab_dict)

print("-" * 50)
print(f"Готово! Все аннотации успешно сохранены в файл: {out_mat_file}")