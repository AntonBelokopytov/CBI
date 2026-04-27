# -*- coding: utf-8 -*-
"""
Created on Wed Jan 21 19:29:56 2026

@author: ansbel
"""

import mne
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec


# %%
import mne
import numpy as np
import os
import glob
from scipy.io import savemat

# 1. Настройка путей
data_dir = 'D:/OS(CURRENT)/data/parkinson/control/'
# Ищем все файлы типа Control_..._epochs.fif
file_pattern = os.path.join(data_dir, 'Control_*_CenterOut_epochs.fif')

# Папка, куда сохраним списки индексов
out_dir = 'D:/OS(CURRENT)/data/parkinson/indices_simple/'
os.makedirs(out_dir, exist_ok=True)

subject_files = glob.glob(file_pattern)

for dpath in subject_files:
    # Определяем имя испытуемого (например, Control_3)
    basename = os.path.basename(dpath)
    sub_name = basename.split('_CenterOut')[0]
    
    print(f"Обработка {sub_name}...")
    
    # Загружаем только метаданные (preload=False), чтобы было мгновенно
    epochs = mne.read_epochs(dpath, preload=False, verbose=False)
    
    if epochs.metadata is not None:
        # Применяем твою маску
        mask = (
            (epochs.metadata['pp'] == 2) & 
            (epochs.metadata['correct_trials'] == 1)
        )
        
        # Получаем индексы
        # ВАЖНО: прибавляем 1, если планируешь использовать в MATLAB!
        # Если только в Python — убавь "+ 1"
        idx = np.where(mask)[0] + 1 
        
        # --- Вариант А: Сохранение в .mat файл (для MATLAB) ---
        savemat(os.path.join(out_dir, f"{sub_name}_idx.mat"), {'idx': idx})
        
        # --- Вариант Б: Сохранение в .txt (простой текстовый список) ---
        # np.savetxt(os.path.join(out_dir, f"{sub_name}_idx.txt"), idx, fmt='%d')
        
        print(f"  Найдено индексов: {len(idx)}. Сохранено в {sub_name}_idx.mat")
    else:
        print(f"  У {sub_name} отсутствуют метаданные.")

print("\nГотово! Все списки сохранены.")

# %%



# %%
# For parkinson example chose 8-12 Hz filter below
# dpath = 'D:/OS(CURRENT)/data/parkinson/pathology/Patient_1_CenterOut_OFF_EEG_clean_epochs.fif'
dpath = 'D:/OS(CURRENT)/data/parkinson/control/Control_3_CenterOut_epochs.fif'
# dpath = 'D:/OS(CURRENT)/data/parkinson/control/Control_3_CenterOut_epochs.fif'

epochs = mne.read_epochs(dpath, preload=True)

epochs.info

# %%
epochs_eeg = epochs.copy().pick_channels(epochs.ch_names[:38])
epochs_vars = epochs.copy().pick_channels(epochs.ch_names[38:])

# %%
freqs = np.arange(2, 30, 1)

# Количество циклов для вейвлета. Динамическое значение (freqs / 2.0) 
# дает лучшее разрешение по времени для низких частот и по частоте для высоких.
n_cycles = freqs / 2.0 

# 2. Рассчитываем TFR
# average=True усреднит данные по всем эпохам. return_itc=False отключает расчет межгрупповой когерентности.
power = mne.time_frequency.tfr_morlet(epochs_eeg, 
                                      freqs=freqs, 
                                      n_cycles=n_cycles, 
                                      return_itc=False, 
                                      average=True)

# %%
import matplotlib.pyplot as plt

# Создаем копию, чтобы случайно не применить бейзлайн несколько раз при перезапуске ячейки
power_bsl = power.copy()

# Применяем коррекцию базовой линии ко всем данным (от начала эпохи до 0)
# mode='percent' переведет значения в процентное изменение (ERD/ERS)
power_bsl.apply_baseline(baseline=(None, 0), mode='percent')

# ==============================================================================
# 1. Средняя мощность по всем отведениям в мю (8-12 Гц) и бете (13-29 Гц)
# ==============================================================================

# Создаем график
fig, ax = plt.subplots(figsize=(10, 5))

# Вырезаем нужные диапазоны частот с помощью crop()
# Свойство .data возвращает массив NumPy размерности (каналы, частоты, время).
# Усредняем данные по каналам (axis=0) и частотам (axis=1), оставляя только время.
mu_power = power_bsl.copy().crop(fmin=8, fmax=12).data.mean(axis=(0, 1))
beta_power = power_bsl.copy().crop(fmin=13, fmax=29).data.mean(axis=(0, 1))

# Вектор времени берем напрямую из объекта power
time_axis = power_bsl.times

# Строим линии
ax.plot(time_axis, mu_power, label='Мю-ритм (8-12 Гц)', linewidth=2, color='#1f77b4')
ax.plot(time_axis, beta_power, label='Бета-ритм (13-29 Гц)', linewidth=2, color='#d62728')

# Оформление графика
ax.axvline(0, color='black', linestyle='--', label='Событие (t=0)')
ax.axhline(0, color='gray', linestyle='-') # Линия нулевого изменения
ax.set_title('Средняя динамика мощности (ERD/ERS) по всем отведениям')
ax.set_xlabel('Время (с)')
ax.set_ylabel('Изменение мощности (%)')
ax.legend()
ax.grid(True, alpha=0.3)

plt.show()

# ==============================================================================
# 2. Топография с вложенными частотно-временными графиками
# ==============================================================================

# В MNE для этого есть специальный метод plot_topo().
# Так как мы уже применили apply_baseline(), данные отрисуются сразу с учетом коррекции.
# Если у ваших данных нет стандартных координат электродов (montage), 
# MNE может потребовать их задать через epochs_eeg.set_montage('standard_1020')

# %%
fig_topo = power_bsl.plot_topo(title='Топография TFR', 
                               # vmin=-1.5, vmax=1.5, # Лимиты цвета (подгоните под ваши данные)
                               fig_facecolor='w', 
                               font_color='k')
plt.show()

# %%
import numpy as np

# Убедимся, что у нас есть нужные каналы для ROI (Region of Interest)
# Замените названия, если в вашей системе они пишутся иначе (например, 'EEG C3')
# motor_channels = ['C3', 'C4', 'Cz']
motor_channels = ['C3']
picks_motor = [ch for ch in motor_channels if ch in power_bsl.ch_names]

if not picks_motor:
    print("Внимание: каналы C3, C4, Cz не найдены. Берутся первые 3 канала для демонстрации.")
    picks_motor = power_bsl.ch_names[:3]

# ==============================================================================
# 3. TFR Heatmap (Тепловая карта) для сенсомоторной зоны
# ==============================================================================
# Усредняем данные по выбранным каналам (combine='mean'), чтобы получить чистый паттерн ERD/ERS

# fig_tfr = power_bsl.copy().pick_channels(picks_motor).plot(
#     combine='mean',
#     title=f'Усредненный TFR для сенсомоторной коры ({", ".join(picks_motor)})',
#     # vmin=-2.0, vmax=2.0,  # Лимиты шкалы (в процентах). Подгоните при необходимости.
#     cmap='RdBu_r'         # Красно-синяя палитра (синий - десинхронизация/ERD, красный - синхронизация/ERS)
# )

# ==============================================================================
# 4. Топокарты (Topomaps) для конкретных окон (Мю и Бета)
# ==============================================================================
# Строим топографии распределения мощности для конкретных окон после стимула.
# Допустим, нас интересует окно от 0.0 до 1.0 секунды.

fig, axes = plt.subplots(1, 3, figsize=(10, 5))

axes[0].set_title('Мю (9-14 Гц)\n-1.0 - 0.0 с')
power_bsl.plot_topomap(
    tmin=-1.0, tmax=0.0, fmin=9, fmax=14,
    axes=axes[0], show=False, contours=0,
    # vmin=-1.5, vmax=1.5, 
    cmap='RdBu_r', colorbar=True
)

axes[1].set_title('Мю (9-14 Гц)\n0.0 - 1.0 с')
# Мю-ритм (8-12 Гц)
power_bsl.plot_topomap(
    tmin=0.0, tmax=1.0, fmin=9, fmax=14,
    axes=axes[1], show=False, contours=0,
    # vmin=-1.5, vmax=1.5, 
    cmap='RdBu_r', colorbar=True
)

axes[2].set_title('Мю (9-14 Гц)\n2.0 - 3.0 с')
# Бета-ритм (13-29 Гц)
power_bsl.plot_topomap(
    tmin=1.0, tmax=2.0, fmin=9, fmax=14,
    axes=axes[2], show=False, contours=0,
    # vmin=-1.5, vmax=1.5, 
    cmap='RdBu_r', colorbar=True
)

fig.suptitle('Пространственное распределение ERD/ERS (%)', fontsize=14)
plt.show()

# ==============================================================================
# 5. Joint Plot (Комбинированный график TFR + Topo)
# ==============================================================================
# Этот график сам находит пики изменений и рисует для них топографии.
# timefreqs: можно задать конкретные точки [(время, частота), ...], 
# либо оставить None, и MNE найдет локальные экстремумы автоматически.

# Для примера ограничим график частотами до 30 Гц и зададим пару точек вручную (или автоматически)
fig_joint = power_bsl.plot_joint(
    timefreqs=[(0.5, 20), (2.5, 20), (3, 20)], # Смотрим топографию на 0.5с (10 Гц) и 1.5с (20 Гц)
    title='C3, C4, Cz time-frequency',
    # vmin=-2.0, vmax=2.0,
    cmap='RdBu_r'
)
plt.show()

# %%
# 3. Настройка параметров отрисовки
# Мю-ритм лучше всего виден на электродах сенсомоторной зоны (С3, С4, Cz).
sensorimotor_channels = ['C3']
# Проверяем, есть ли эти каналы в ваших данных
picks = [ch for ch in sensorimotor_channels if ch in epochs_eeg.ch_names]

if not picks:
    print("Стандартные сенсомоторные каналы не найдены. Будет отрисован первый доступный канал.")
    picks = [epochs_eeg.ch_names[0]]

# Задаем базовую линию (baseline). 
# Например, от начала эпохи до момента 0 (момент наступления события). 
# Замените значения на те, которые подходят для вашего дизайна эксперимента!
baseline = (None, 0) 

# 4. Строим график
# mode='percent' покажет процентное изменение мощности относительно базовой линии
fig = power.plot(picks=picks, 
                 baseline=baseline, 
                 mode='percent', 
                 title='C3',
                 ) # Лимиты цветовой шкалы (возможно, придется подогнать под ваши данные)

plt.show()

# %%
# 1. Извлекаем монтаж
montage = epochs.copy().get_montage()
pos = montage.get_positions()['ch_pos']

# 2. Смещаем все координаты на 2 см назад по оси Y
for ch in pos:
    pos[ch][1] -= 0.02  # Сдвиг по Y на -20мм

# 3. Пересоздаем монтаж с новыми координатами
new_montage = mne.channels.make_dig_montage(ch_pos=pos, coord_frame='head')

# 4. Применяем его к данным
epochs_eeg.set_montage(new_montage)

# Теперь рисуется корректно по умолчанию
epochs_eeg.plot_sensors(show_names=True)

# %%
mask = (
    (epochs.metadata['pp'] == 2) &
    (epochs.metadata['correct_trials'] == 1)
)

idx = np.where(mask)[0]
idx

# %%
epochs_eeg[idx].copy().filter(l_freq=0.1,h_freq=20).average(tmin=2.8,tmax=4).plot()

# %%
# Вариант с обрезкой до усреднения
epochs_eeg[idx].copy().filter(0.1, 20).crop(tmin=1.8,tmax=3).average().plot()

# %%
epochs_eeg[idx].compute_psd(fmin=1,fmax=35).plot()

# %%
epochs_pp2 = epochs[idx]

cond1 = epochs_pp2.metadata['cond'] == 1
cond3 = epochs_pp2.metadata['cond'] == 3

# %%
X = epochs_pp2.get_data()[:,:38]
Fs = epochs_pp2.info['sfreq']
fmin = 2
fmax = 8

from scipy.signal import butter, filtfilt
band = [fmin, fmax]
b, a = butter(4, np.array(band) / (Fs / 2), btype='band')
Xfilt = filtfilt(b, a, X, axis=-1)

# Xfilt = Xfilt[:,:,:3000]

import src.env_corrca as env_functions
W, A, corrs, Epochs_cov, eigenvalues = env_functions.env_corrca(Xfilt,Fs)

# %%
# Check eigenvalues and correlations!
# Opposite components might matter: according to the
# absolute value of eigenvalue OR InterTrial correlation
plt.figure()

plt.stem(
    eigenvalues,
    linefmt='b-',
    markerfmt='bo',
    basefmt=" "
)

plt.stem(
    corrs,
    linefmt='r-',
    markerfmt='ro',
    basefmt=" "
)

plt.xlabel('num of component')
plt.legend([
    'Eigenvalue',
    'InterTrial correlation'
])

plt.show()

# %%
com_idx = 35
w = W[:,com_idx]
a = A[:,com_idx]
    
# Лучше сделать максимальное абсолютное значение паттерна положительным
imax = np.argmax(np.abs(a))
sign_flip = np.sign(a[imax])
a = a * sign_flip

fig, axes = plt.subplots(1, 2, figsize=(8, 4), facecolor='white')

# -------- Filter --------
mne.viz.plot_topomap(
    w,
    epochs_eeg.info,
    axes=axes[0],
    names=epochs_eeg.ch_names,   
    show=True,
    cmap='RdBu_r',
)

axes[0].set_title('Filter')

# -------- Pattern --------
mne.viz.plot_topomap(
    a,
    epochs_eeg.info,
    axes=axes[1],
    names=epochs_eeg.ch_names,   
    show=True,
    cmap='RdBu_r',
)

axes[1].set_title('Pattern')

plt.tight_layout()
plt.show()

# %% GET ANALYTIC SIGNAL
from scipy.signal import hilbert
com_idx = 0
w = W[:,com_idx]
a = A[:,com_idx]

# Лучше сделать максимальное абсолютное значение паттерна положительным
imax = np.argmax(np.abs(a))
sign_flip = np.sign(a[imax])
a = a * sign_flip

Xcomp = w @ np.permute_dims(Xfilt,(1,0,2)).reshape([38,-1])
Xcomp = Xcomp.reshape([Xfilt.shape[0],Xfilt.shape[2]])

analytic = hilbert(Xcomp, axis=0)
envelope = np.abs(analytic)


env1 = envelope[cond1]
env3 = envelope[cond3]

mean1 = env1.mean(axis=0)
std1 = env1.std(axis=0)

mean3 = env3.mean(axis=0)
std3 = env3.std(axis=0)

t = np.arange(mean1.shape[0]) / Fs

plt.figure()

plt.plot(t, mean1, label='cond1')
plt.fill_between(t, mean1-std1, mean1+std1, alpha=0.3)

plt.plot(t, mean3, label='cond3')
plt.fill_between(t, mean3-std3, mean3+std3, alpha=0.3)

plt.legend()
plt.xlabel('Time (s)')
plt.ylabel('Envelope')
plt.show()

# %% 
# Keep in mind that the graphs show edge effects.
fig, axes = plt.subplots(1, 2, figsize=(8, 4), facecolor='white')

# -------- Filter --------
mne.viz.plot_topomap(
    w,
    epochs_eeg.info,
    axes=axes[0],
    names=epochs_eeg.ch_names,   
    show=True,
    cmap='RdBu_r',
)

axes[0].set_title('Filter')

# -------- Pattern --------
mne.viz.plot_topomap(
    a,
    epochs_eeg.info,
    axes=axes[1],
    names=epochs_eeg.ch_names,   
    show=True,
    cmap='RdBu_r',
)

axes[1].set_title('Pattern')

plt.tight_layout()
plt.show()


n_epochs, T = envelope.shape

# --- время в секундах ---
time_sec = np.arange(T) / Fs
t_min = time_sec[0]
t_max = time_sec[-1]

mean_envelope = np.mean(envelope, axis=0)
std_envelope  = np.std(envelope, axis=0)

# ---- можно задать верхний предел яркости ----
vmax = 3 * np.max(std_envelope)   # ← меняй при необходимости
vmin = 0

fig = plt.figure(figsize=(10, 8), facecolor='white')
gs = GridSpec(3, 2, width_ratios=[20, 1],
              height_ratios=[2, 1, 1], figure=fig)

# =========================
# 1 Heatmap
# =========================
ax1 = fig.add_subplot(gs[0, 0])
cax = fig.add_subplot(gs[0, 1])

im = ax1.imshow(
    envelope,
    aspect='auto',
    origin='lower',
    extent=[t_min, t_max, 0, n_epochs],
    cmap='viridis',
    vmin=vmin,
    vmax=vmax
)

fig.colorbar(im, cax=cax)

ax1.set_ylabel('Epoch')
ax1.set_title('Envelope per epoch')

# =========================
# 2 All epochs (raw)
# =========================
ax2 = fig.add_subplot(gs[1, 0], sharex=ax1)

ax2.plot(time_sec, Xcomp.T, alpha=0.5)
ax2.set_ylabel('Amplitude')
ax2.set_title('Component activity (all epochs)')
ax2.grid(True)

# =========================
# 3 Mean envelope ± SD
# =========================
ax3 = fig.add_subplot(gs[2, 0], sharex=ax1)

ax3.fill_between(
    time_sec,
    mean_envelope - std_envelope,
    mean_envelope + std_envelope,
    alpha=0.2
)

ax3.plot(time_sec, mean_envelope, linewidth=2)

ax3.set_ylabel('Envelope')
ax3.set_xlabel('Time (s)')
ax3.set_title('Mean envelope ± SD')
ax3.grid(True)

plt.tight_layout()
plt.show()
