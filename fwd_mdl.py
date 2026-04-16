import mne
import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import savemat

# %%
# 1. Загрузка fsaverage
fs_dir = mne.datasets.fetch_fsaverage(verbose=False)
bem_fname = os.path.join(fs_dir, 'bem', 'fsaverage-5120-5120-5120-bem-sol.fif')
src_fname = os.path.join(fs_dir, 'bem', 'fsaverage-ico-5-src.fif')
trans = 'fsaverage' 

# 2. Каналы и маппинг
my_channels = [
    'Fp1', 'Fpz', 'Fp2', 'F7', 'F3', 'Fz', 'F4', 'F8', 'FT7', 'FC3', 'FCz',
    'FC4', 'FT8', 'T3', 'C3', 'Cz', 'C4', 'T4', 'TP7', 'CP3', 'CPz', 'CP4',
    'TP8', 'T5', 'P3', 'Pz', 'P4', 'T6', 'P5', 'PO3', 'POz', 'PO4', 'P6',
    'PO7', 'O1', 'Oz', 'O2', 'PO8'
]

ch_mapping = {'T3': 'T7', 'T4': 'T8', 'T5': 'P7', 'T6': 'P8'}
mne_channels = [ch_mapping.get(ch, ch) for ch in my_channels]

# 3. Info и монтаж
info = mne.create_info(ch_names=mne_channels, sfreq=1000., ch_types='eeg')
montage = mne.channels.make_standard_montage('standard_1005')
info.set_montage(montage)
info.rename_channels({v: k for k, v in ch_mapping.items()})

# %%
# 4. Расчет forward модели
print("Рассчитываем forward solution...")
fwd = mne.make_forward_solution(
    info=info,
    trans=trans,
    src=src_fname,
    bem=bem_fname,
    eeg=True,
    meg=False,
    mindist=5.0,
    n_jobs=-1
)

# %%
# 5. 
fwd_fixed = mne.convert_forward_solution(fwd, surf_ori=False, force_fixed=False, use_cps=True)

# 6. Извлекаем матрицу лидфилда и координаты
leadfield_matrix = fwd_fixed['sol']['data']

# ВАЖНО: fwd_fixed['source_rr'] содержит точные 3D-координаты всех 20484 диполей
source_positions = fwd_fixed['source_rr'] 

# Извлекаем 3D-координаты сенсоров из структуры Info
sensor_positions = np.array([ch['loc'][:3] for ch in info['chs']])

print("-" * 30)
print(f"Размерность матрицы смешивания (A): {leadfield_matrix.shape}")
print("-" * 30)

# %%
# =====================================================================
# 6.5 БЕЗОПАСНАЯ 3D-ВИЗУАЛИЗАЦИЯ ЧЕРЕЗ MATPLOTLIB
# =====================================================================
print("Запуск 3D-визуализации через Matplotlib...")

fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

# Рисуем источники (мозг) мелкими серыми точками
ax.scatter(source_positions[:, 0], source_positions[:, 1], source_positions[:, 2], 
           c='gray', s=1, alpha=0.05, label='Brain Sources (Dipoles)')

# Рисуем электроды (сенсоры) крупными красными точками
ax.scatter(sensor_positions[:, 0], sensor_positions[:, 1], sensor_positions[:, 2], 
           c='red', s=50, depthshade=True, label='EEG Sensors')

# Добавляем текстовые подписи для каналов
for i, ch_name in enumerate(info.ch_names):
    ax.text(sensor_positions[i, 0], sensor_positions[i, 1], sensor_positions[i, 2], 
            ch_name, fontsize=8, zorder=10, color='black')

ax.set_title("Корегистрация: Сенсоры ЭЭГ и Источники fsaverage")
ax.legend()
ax.axis('off') # Отключаем оси X, Y, Z для красоты
plt.tight_layout()
plt.show()
# =====================================================================

# %%
leadfield_matrix = fwd_fixed['sol']['data']
source_positions = fwd_fixed['source_rr'] 

# ВАЖНО: извлекаем векторы ориентации (нормали)
# Для фиксированной модели это единичные векторы [N_sources x 3]
source_orientations = fwd_fixed['source_nn']

# Извлекаем 3D-координаты сенсоров
sensor_positions = np.array([ch['loc'][:3] for ch in info['chs']])

# 7. Экспорт в .mat файл
savemat('fsaverage_38ch_leadfield.mat', {
    'A': leadfield_matrix,           
    'source_pos': source_positions,  
    'source_ori': source_orientations, # Добавляем ориентации
    'sensor_pos': sensor_positions,  
    'channels': my_channels          
})

print(f"Модель сохранена. Ориентации: {source_orientations.shape}")

# %%