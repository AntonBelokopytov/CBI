import numpy as np
import mne
from mne.datasets import fetch_fsaverage, eegbci
from mne.io.constants import FIFF
from scipy.io import savemat

# %%
# =====================================================================
# 1. Загрузка анатомии fsaverage
# =====================================================================
fs_dir = fetch_fsaverage(verbose=True)

subject = "fsaverage"
trans = "fsaverage"  
src = fs_dir / "bem" / "fsaverage-ico-5-src.fif"
bem = fs_dir / "bem" / "fsaverage-5120-5120-5120-bem-sol.fif"

# %%
# =====================================================================
# 2. Загрузка 64-канальных данных (согласно туториалу MNE)
# =====================================================================
print("Загрузка данных BCI...")
(raw_fname,) = eegbci.load_data(subjects=1, runs=[6])
raw = mne.io.read_raw_edf(raw_fname, preload=True)

# Очистка имен каналов для совместимости со стандартом 10-05
eegbci.standardize(raw)
montage = mne.channels.make_standard_montage("standard_1005")

# Применяем монтаж и устанавливаем референс
raw.set_montage(montage)
raw.set_eeg_reference(projection=True) 

n_channels = len(raw.ch_names)
print(f"Количество каналов после загрузки: {n_channels}")

# %%
# =====================================================================
# 3. Расчет прямой модели (Forward Solution)
# =====================================================================
print("Рассчитываем forward solution для 64 каналов...")
fwd = mne.make_forward_solution(
    raw.info, 
    trans=trans, 
    src=src, 
    bem=bem, 
    eeg=True, 
    meg=False, 
    mindist=5.0, 
    n_jobs=-1  
)

# Снимаем ограничения ориентации для сохранения полных 3D-векторов диполей
fwd_fixed = mne.convert_forward_solution(fwd, surf_ori=False, force_fixed=False, use_cps=True)

# %%
# =====================================================================
# 4. Извлечение матриц и геометрии
# =====================================================================
leadfield_matrix = fwd_fixed['sol']['data']
source_positions = fwd_fixed['source_rr']
source_orientations = fwd_fixed['source_nn']
sensor_positions = np.array([ch['loc'][:3] for ch in raw.info['chs']])

# %%
# =====================================================================
# 5. Подготовка структуры elec для MATLAB (FieldTrip)
# =====================================================================
# Переводим сенсоры из метров в миллиметры
sensor_positions_mm = sensor_positions * 1000.0

# Формируем cell array для названий каналов
labels_cell = np.empty((n_channels, 1), dtype=object)
for i, ch in enumerate(raw.info.ch_names):
    labels_cell[i, 0] = ch

# Извлекаем фидуциальные точки (Nasion, LPA, RPA)
fid_labels = []
fid_pos = []

if raw.info.get('dig') is not None:
    for d in raw.info['dig']:
        if d['kind'] == FIFF.FIFFV_POINT_CARDINAL:
            if d['ident'] == FIFF.FIFFV_POINT_LPA:
                fid_labels.append('lpa')
                fid_pos.append(d['r'])
            elif d['ident'] == FIFF.FIFFV_POINT_NASION:
                fid_labels.append('nasion')
                fid_pos.append(d['r'])
            elif d['ident'] == FIFF.FIFFV_POINT_RPA:
                fid_labels.append('rpa')
                fid_pos.append(d['r'])

if fid_pos:
    fid_pos_mm = np.array(fid_pos) * 1000.0
else:
    fid_pos_mm = np.zeros((3, 3))

fid_labels_cell = np.empty((len(fid_labels), 1), dtype=object)
for i, lab in enumerate(fid_labels):
    fid_labels_cell[i, 0] = lab

# Вложенная структура fid
fid_struct = {
    'pos': fid_pos_mm,
    'label': fid_labels_cell,
    'unit': 'mm'
}

# Основная структура elec
elec_struct = {
    'label': labels_cell,
    'chanpos': sensor_positions_mm,
    'elecpos': sensor_positions_mm,
    'tra': np.eye(n_channels),
    'unit': 'mm',
    'type': 'eeg',
    'fid': fid_struct
}

# %%
# =====================================================================
# 6. Экспорт в .mat
# =====================================================================
output_filename = 'fsaverage_64ch_leadfield.mat'
savemat(output_filename, {
    'A': leadfield_matrix,           
    'source_pos': source_positions,  
    'source_ori': source_orientations, 
    'sensor_pos': sensor_positions,  
    'channels': labels_cell,         
    'elec': elec_struct              
})

print("-" * 40)
print(f"Все данные успешно сохранены в {output_filename}")
print(f"Размерность матрицы смешивания (A): {leadfield_matrix.shape}")
print("-" * 40)