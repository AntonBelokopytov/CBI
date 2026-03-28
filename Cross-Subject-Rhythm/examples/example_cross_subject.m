% Скрипт-пример для генерации и поиска индуцированной ритмической активности
% с одинаковой огибающей, но разной фазой и локализацией для группы субъектов.

clear; close all; clc;

% Добавляем пути к исходникам
addpath(fullfile(pwd, '..', 'src'));

%% 1. Параметры симуляции
Fs = 250;               % Частота дискретизации, Гц
Ts = 60;                % Длительность записи, с (увеличено для более длинной проверки)
flanker = 1;            % Отступы по краям для устранения краевых эффектов
N_subj = 20;            % Количество субъектов (например, 20 участников)
N_distr = 3;            % Количество независимых целевых индуцированных источников (огибающих)
N_bg = 10;              % Количество фоновых шумящих источников
SNR = 5;                % Дисперсия целевых огибающих
env_noise = 0.1;        % Уровень индивидуального шума огибающей (уменьшен)

% Создаем синтетическую матрицу Leadfield (G)
% Для примера возьмем 30 сенсоров и 100 возможных диполей в мозге (по 3 оси на каждый)
Nsens = 30;
Nsites = 100;
G = randn(Nsens, Nsites * 3);

fprintf('Генерация данных для %d субъектов (%d коррелирующих огибающих)...\n', N_subj, N_distr);
tic;
[X_all_subjects, target_envs] = generate_cross_subject_data(G, N_subj, N_distr, N_bg, Fs, Ts, flanker, SNR, env_noise);
toc;

fprintf('Размерность сгенерированных данных: %d сенсоров x %d отсчетов x %d субъектов\n', ...
    size(X_all_subjects, 1), size(X_all_subjects, 2), size(X_all_subjects, 3));

%% 2. Фильтрация данных (в целевой полосе, например 8-12 Гц для альфа-ритма)
fprintf('Фильтрация данных в полосе 8-12 Гц...\n');
[b, a] = butter(3, [8, 12] / (Fs / 2));
X_filt = zeros(size(X_all_subjects));

for s = 1:N_subj
    % Фильтруем данные каждого субъекта (с нулевым фазовым сдвигом)
    X_filt(:,:,s) = filtfilt(b, a, X_all_subjects(:,:,s)')';
end

%% 3. Поиск индуцированной огибающей через консенсусный граф (env_laplace_dec2)
Wsize = 1;      % Размер окна, с
Ssize = 0.2;    % Шаг окна, с
N_neigb = 10;   % Количество ближайших соседей (k)
lambda = 1e-6;  % Коэффициент регуляризации ковариаций

fprintf('Поиск индуцированной огибающей...\n');
tic;
[z, L, D_graph] = find_induced_envelope_laplace2(X_filt, Fs, Wsize, Ssize, N_neigb, lambda);
toc;

% Огибающая z содержит все найденные компоненты.
% Поскольку алгоритм слепой, порядок извлеченных компонент (столбцов z)
% не обязательно совпадает с порядком строк target_envs.
% Поэтому мы вычислим матрицу кросс-корреляций для сопоставления.

%% 4. Визуализация и сравнение
% Поскольку извлеченная огибающая имеет более низкую частоту дискретизации
% (она оценена по окнам, а не по отсчетам), нам нужно проредить оригинальные
% target_envs для корректного сравнения

window_samples = round(Wsize * Fs);
step_samples = round(Ssize * Fs);
N_epochs = size(z, 1);
target_envs_downsampled = zeros(N_distr, N_epochs);

for d = 1:N_distr
    for i = 1:N_epochs
        start_idx = (i-1)*step_samples + 1;
        end_idx = start_idx + window_samples - 1;
        target_envs_downsampled(d, i) = mean(target_envs(d, start_idx:end_idx));
    end
end

% Удаляем первые несколько окон для устранения краевых эффектов фильтрации и расчета
cut_idx = 5;
target_envs_downsampled_cut = target_envs_downsampled(:, cut_idx:end);
z_cut = z(cut_idx:end, :);

% Нормализуем после обрезки
for d = 1:N_distr
    target_envs_downsampled_cut(d,:) = (target_envs_downsampled_cut(d,:) - mean(target_envs_downsampled_cut(d,:))) / std(target_envs_downsampled_cut(d,:));
end
for c = 1:size(z_cut, 2)
    z_cut(:,c) = (z_cut(:,c) - mean(z_cut(:,c))) / std(z_cut(:,c));
end

% Сопоставление компонент (жадный поиск по максимальной абсолютной корреляции)
matched_z_indices = zeros(1, N_distr);
matched_corrs = zeros(1, N_distr);
matched_signs = ones(1, N_distr);

available_z_indices = 1:size(z_cut, 2);

for d = 1:N_distr
    best_corr = -1;
    best_idx_in_avail = -1;
    best_sign = 1;

    for idx_idx = 1:length(available_z_indices)
        z_idx = available_z_indices(idx_idx);
        c_val = corr(target_envs_downsampled_cut(d,:)', z_cut(:, z_idx));
        if abs(c_val) > best_corr
            best_corr = abs(c_val);
            best_idx_in_avail = idx_idx;
            best_sign = sign(c_val);
        end
    end

    matched_z_indices(d) = available_z_indices(best_idx_in_avail);
    matched_corrs(d) = best_corr;
    matched_signs(d) = best_sign;

    % Удаляем найденный индекс из доступных (один к одному)
    available_z_indices(best_idx_in_avail) = [];
end

time_axis_windows = ((cut_idx-1):N_epochs-1) * Ssize + (Wsize/2);

figure('Name', 'Сравнение нескольких огибающих', 'Position', [100, 100, 800, 200 * N_distr]);

for d = 1:N_distr
    subplot(N_distr, 1, d);

    plot(time_axis_windows, target_envs_downsampled_cut(d,:), 'k', 'LineWidth', 2, 'DisplayName', sprintf('Оригинальная огибающая %d', d));
    hold on;

    matched_z = z_cut(:, matched_z_indices(d)) * matched_signs(d);
    plot(time_axis_windows, matched_z, 'r--', 'LineWidth', 2, 'DisplayName', sprintf('LPP компонента (z_{%d}), r = %.2f', matched_z_indices(d), matched_corrs(d)));

    title(sprintf('Поиск %d-й компоненты (SNR=%.1f, Noise=%.1f)', d, SNR, env_noise));
    ylabel('Амплитуда (z)');
    legend;
    grid on;

    if d == N_distr
        xlabel('Время (с)');
    end
end

print('multi_envelope_comparison.png', '-dpng', '-r300');
disp('Готово! График сохранен в multi_envelope_comparison.png');
