import mne
import numpy as np
import mat73
import matplotlib.pyplot as plt
from mne.preprocessing import ICA
from matplotlib.gridspec import GridSpec

# %%
# 1. Укажи правильный путь (тот, что работал в первый раз)
file_path = 'data/high_dens_eeg/data_bobrov/data_64_64.mat'

print(f"Загрузка файла: {file_path}")
mat = mat73.loadmat(file_path)

# 3. Теперь достаем поля
ch_names = mat['chan_labels']
ch_names = [ch[0] for ch in ch_names]

sfreq = float(mat['samplerate'])
data = mat['data']


info = mne.create_info(ch_names=ch_names, sfreq=sfreq, ch_types='eeg')
raw = mne.io.RawArray(data.T, info) 

# %%
# Делаем из вектора состояний Stim-канал
stim_data = np.array(mat['state']).reshape(1, -1)
stim_info = mne.create_info(['STI'], sfreq, ['stim'])
stim_raw = mne.io.RawArray(stim_data, stim_info)
raw.add_channels([stim_raw])

pos_3d = mat['chan_positions_3d']
ch_pos = {name: pos for name, pos in zip(ch_names, pos_3d)}
montage = mne.channels.make_dig_montage(ch_pos=ch_pos)
raw.set_montage(montage)

# %%
raw.filter(l_freq=0.5, h_freq=70.0)
raw.notch_filter(freqs=50.0) 

# %%
raw.plot(duration=10, scalings='auto')

# %%
matrix_ch = [ch for ch in raw.ch_names if ch.startswith("Matrix")]
matrix_ch.append('STI')
raw_m = raw.copy().pick_channels(matrix_ch)

# %%
raw_m.plot()

# %%
raw.compute_psd(fmin=0.1,fmax=30).plot()

# %%
raw.plot_sensors(kind='3d')

# %%
state_arr = raw.get_data(picks='STI')[0]

change_idx = np.where(np.diff(state_arr) != 0)[0] + 1
change_idx = np.append(change_idx, 55121)
change_idx = np.sort(change_idx)
change_idx

# %%
events_custom = np.column_stack([
    change_idx,
    np.zeros_like(change_idx, dtype=int),
    state_arr[change_idx].astype(int)
])

event_id = {
    'rest': 1,
    'index': 2,
    'middle': 3,
    'ring': 4,
    'pinky': 5
}

epochs = mne.Epochs(raw_m, events_custom, event_id=event_id, 
                    tmin=-1, tmax=11, 
                    baseline=(0, 0),    
                    preload=True,
                    on_missing='warn')

print(epochs)

# %%
epochs.plot(n_epochs=1)

# %%
epochs.drop_channels(epochs_cl.info['bads'])

# %%
from scipy.io import savemat
import numpy as np

epochs.drop_channels(epochs.info['bads'])

data = epochs.get_data()

# координаты каналов
ch_pos = np.array([ch['loc'][:3] for ch in epochs.info['chs']])

# стимулы
events = epochs.events
labels = events[:, 2]

# словарь соответствия классов
event_id = epochs.event_id

mat_dict = {
    "epochs": data,
    "labels": labels,
    "events": events,
    "event_id": event_id,
    "ch_pos": ch_pos,
    "ch_names": np.array(epochs.ch_names, dtype=object),
    "times": epochs.times,
    "sfreq": epochs.info["sfreq"]
}

savemat("epochs_dataset.mat", mat_dict)

# %%
epochs.plot(n_epochs=1, scalings='auto')

# %%
epochs_cl = epochs.copy().filter(l_freq=0.5, h_freq=40.0)

# %%
epochs_cl.plot(n_epochs=1, scalings='auto')

# %%
ica = ICA(n_components=0.9999, method='fastica', random_state=42, max_iter='auto')
ica.fit(epochs)

# %%
ica.plot_sources(epochs)

# %%
clear_epochs = ica.apply(epochs)

# %%
clear_epochs.plot()

# %%
epochs['index'].compute_psd(fmin=0.1,fmax=70).plot()

# %%
Xrest = epochs['rest'].get_data()[:, :-1, :]
Xindex = epochs['index'].get_data()[:, :-1, :]
Xmiddle = epochs['middle'].get_data()[:, :-1, :]
Xring = epochs['ring'].get_data()[:, :-1, :]
Xpinky = epochs['pinky'].get_data()[:, :-1, :]

X = []
# Проходимся по 10 эпохам
for i in range(9):
    # Достаем конкретную i-тую эпоху [128 x время]
    rest_i = Xrest[i]
    index_i = Xindex[i]
    middle_i = Xmiddle[i]
    ring_i = Xring[i]
    pinky_i = Xpinky[i]
    
    # Склеиваем их по оси времени (axis=1 для 2D массивов)
    concat_epoch = np.concatenate([rest_i, index_i, middle_i, ring_i, pinky_i], axis=1)
    
    X.append(concat_epoch)

# Переводим обратно в удобный 3D-массив numpy [10 x 128 x общее_время]
X = np.array(X)
print("Итоговая размерность X:", X.shape)

# %%
Fs = epochs.info['sfreq']
fmin = 8
fmax = 12

from scipy.signal import butter, filtfilt
band = [fmin, fmax]
b, a = butter(4, np.array(band) / (Fs / 2), btype='band')
Xfilt = filtfilt(b, a, X, axis=-1)

# %%
import src.env_corrca as env_functions
W, A, corrs, Epochs_cov, eigenvalues = env_functions.env_corrca(Xfilt,Fs)

# %%
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
from scipy.signal import hilbert
import matplotlib.pyplot as plt

com_idx = 0
w = W[:, com_idx]
a = A[:, com_idx]

# Лучше сделать максимальное абсолютное значение паттерна положительным
imax = np.argmax(np.abs(a))
sign_flip = np.sign(a[imax])
a = a * sign_flip

# Вычисления Xcomp и огибающей (оставил как у тебя)
Xcomp = w @ np.transpose(Xfilt, (1, 0, 2)).reshape([Xfilt.shape[1], -1])
Xcomp = Xcomp.reshape([Xfilt.shape[0], Xfilt.shape[2]])

analytic = hilbert(Xcomp, axis=0)
envelope = np.abs(analytic)

# ==========================================================
# ДОБЫВАЕМ 3D КООРДИНАТЫ ЭЛЕКТРОДОВ
# ==========================================================
# Достаем словарь с позициями из монтажа
pos_dict = epochs.info.get_montage().get_positions()['ch_pos']

# Выстраиваем координаты строго в том порядке, в котором идут каналы
coords = np.array([pos_dict[ch] for ch in epochs.ch_names[:-1]])
x, y, z = coords[:, 0], coords[:, 1], coords[:, 2]

# Для симметрии цветовой шкалы найдем максимальные по модулю значения
vmax_w = np.max(np.abs(w))
vmax_a = np.max(np.abs(a))

# ==========================================================
# ФИГУРА 1: 3D Scatter (Filter & Pattern)
# ==========================================================
fig1 = plt.figure(figsize=(10, 5), facecolor='white')

# -------- Filter --------
ax1 = fig1.add_subplot(121, projection='3d')
# c=w кодирует цвет, vmin/vmax центрируют белый цвет на 0
sc1 = ax1.scatter(x, y, z, c=w, cmap='RdBu_r', s=60, alpha=0.9, 
                  edgecolors='k', vmin=-vmax_w, vmax=vmax_w)
ax1.set_title('Filter')
fig1.colorbar(sc1, ax=ax1, shrink=0.5, pad=0.1)

# Настройка вида: убираем оси для красоты и задаем угол
ax1.set_axis_off()
ax1.view_init(elev=30, azim=45) 

# -------- Pattern --------
ax2 = fig1.add_subplot(122, projection='3d')
sc2 = ax2.scatter(x, y, z, c=a, cmap='RdBu_r', s=60, alpha=0.9, 
                  edgecolors='k', vmin=-vmax_a, vmax=vmax_a)
ax2.set_title('Pattern')
fig1.colorbar(sc2, ax=ax2, shrink=0.5, pad=0.1)

ax2.set_axis_off()
ax2.view_init(elev=30, azim=45)

plt.tight_layout()
plt.show()

# ==========================================================
# ФИГУРА 2: Огибающие и динамика (твой оригинальный код)
# ==========================================================
n_epochs, T = envelope.shape

# --- время в секундах ---
time_sec = np.arange(T) / Fs
t_min = time_sec[0]
t_max = time_sec[-1]

mean_envelope = np.mean(envelope, axis=0)
std_envelope  = np.std(envelope, axis=0)

vmax_env = 3 * np.max(std_envelope)   
vmin_env = 0

fig2 = plt.figure(figsize=(10, 8), facecolor='white')
gs = GridSpec(3, 2, width_ratios=[20, 1], height_ratios=[2, 1, 1], figure=fig2)

# 1 Heatmap
ax3 = fig2.add_subplot(gs[0, 0])
cax = fig2.add_subplot(gs[0, 1])

im = ax3.imshow(
    envelope,
    aspect='auto',
    origin='lower',
    extent=[t_min, t_max, 0, n_epochs],
    cmap='viridis',
    vmin=vmin_env,
    vmax=vmax_env
)
fig2.colorbar(im, cax=cax)
ax3.set_ylabel('Epoch')
ax3.set_title('Envelope per epoch')

# 2 All epochs (raw)
ax4 = fig2.add_subplot(gs[1, 0], sharex=ax3)
ax4.plot(time_sec, Xcomp.T, alpha=0.5)
ax4.set_ylabel('Amplitude')
ax4.set_title('Component activity (all epochs)')
ax4.grid(True)

# 3 Mean envelope ± SD
ax5 = fig2.add_subplot(gs[2, 0], sharex=ax3)
ax5.fill_between(
    time_sec,
    mean_envelope - std_envelope,
    mean_envelope + std_envelope,
    alpha=0.2
)
ax5.plot(time_sec, mean_envelope, linewidth=2)
ax5.set_ylabel('Envelope')
ax5.set_xlabel('Time (s)')
ax5.set_title('Mean envelope ± SD')
ax5.grid(True)

plt.tight_layout()
plt.show()

# %%
com_idx = 0
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
    epochs.info,
    axes=axes[0],
    # names=clear_epochs.ch_names,   
    show=False,
    cmap='RdBu_r',
)

axes[0].set_title('Filter')

# -------- Pattern --------
mne.viz.plot_topomap(
    a,
    epochs.info,
    axes=axes[1],
    # names=clear_epochs.ch_names,   
    show=False,
    cmap='RdBu_r',
)

axes[1].set_title('Pattern')

plt.tight_layout()
plt.show()

# %%
from scipy.signal import hilbert
com_idx = 0
w = W[:,com_idx]
a = A[:,com_idx]

# Лучше сделать максимальное абсолютное значение паттерна положительным
imax = np.argmax(np.abs(a))
sign_flip = np.sign(a[imax])
a = a * sign_flip

Xcomp = w @ np.permute_dims(Xfilt,(1,0,2)).reshape([64,-1])
Xcomp = Xcomp.reshape([Xfilt.shape[0],Xfilt.shape[2]])

analytic = hilbert(Xcomp, axis=0)
envelope = np.abs(analytic)

# Keep in mind that the graphs show edge effects.
fig, axes = plt.subplots(1, 2, figsize=(8, 4), facecolor='white')

# -------- Filter --------
mne.viz.plot_topomap(
    w,
    epochs.info,
    axes=axes[0],
    # names=clear_epochs.ch_names,   
    show=True,
    cmap='RdBu_r',
)

axes[0].set_title('Filter')

# -------- Pattern --------
mne.viz.plot_topomap(
    a,
    epochs.info,
    axes=axes[1],
    # names=clear_epochs.ch_names,   
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
