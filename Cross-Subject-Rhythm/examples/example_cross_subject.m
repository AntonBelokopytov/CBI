% Скрипт-пример для генерации и поиска индуцированной ритмической активности
% с одинаковой огибающей, но разной фазой и локализацией для группы субъектов.

clear; close all; clc;

% Добавляем пути к исходникам
addpath(fullfile(pwd, '..', 'src'));

%% 1. Параметры симуляции
Fs = 250;               % Частота дискретизации, Гц
Ts = 60;                % Длительность записи, с (увеличено для более длинной проверки)
flanker = 1;            % Отступы по краям для устранения краевых эффектов
N_subj = 10;            % Количество субъектов (например, 10 участников)
N_distr = 1;            % Количество целевых индуцированных источников
N_bg = 10;              % Количество фоновых шумящих источников
SNR = 5;                % Отношение сигнал/шум для амплитуды целевой огибающей

% Создаем синтетическую матрицу Leadfield (G)
% Для примера возьмем 30 сенсоров и 100 возможных диполей в мозге (по 3 оси на каждый)
Nsens = 30;
Nsites = 100;
G = randn(Nsens, Nsites * 3);

fprintf('Генерация данных для %d субъектов...\n', N_subj);
tic;
[X_all_subjects, target_env] = generate_cross_subject_data(G, N_subj, N_distr, N_bg, Fs, Ts, flanker, SNR);
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

% Огибающая z содержит все найденные компоненты
% Первая компонента (первый столбец) - самая сильная общая огибающая
Env_sum = z(:, 1);

%% 4. Визуализация и сравнение
% Поскольку извлеченная огибающая имеет более низкую частоту дискретизации
% (она оценена по окнам, а не по отсчетам), нам нужно проредить оригинальную
% target_env для корректного сравнения

window_samples = round(Wsize * Fs);
step_samples = round(Ssize * Fs);
N_epochs = length(Env_sum);
target_env_downsampled = zeros(1, N_epochs);

for i = 1:N_epochs
    start_idx = (i-1)*step_samples + 1;
    end_idx = start_idx + window_samples - 1;
    target_env_downsampled(i) = mean(target_env(start_idx:end_idx));
end

% Нормализуем для удобства отображения на одном графике
target_env_downsampled = (target_env_downsampled - mean(target_env_downsampled)) / std(target_env_downsampled);

% Удаляем первые несколько окон для устранения краевых эффектов фильтрации и расчета
cut_idx = 5;
target_env_downsampled_cut = target_env_downsampled(cut_idx:end);
Env_sum_cut = Env_sum(cut_idx:end);
N_epochs_cut = length(Env_sum_cut);

% Повторно нормализуем после обрезки
target_env_downsampled_cut = (target_env_downsampled_cut - mean(target_env_downsampled_cut)) / std(target_env_downsampled_cut);
Env_sum_cut = (Env_sum_cut - mean(Env_sum_cut)) / std(Env_sum_cut);

% Знак компоненты может быть перевернут, поэтому проверяем корреляцию
% и при необходимости инвертируем
corr_val = corr(target_env_downsampled_cut', Env_sum_cut);
if corr_val < 0
    Env_sum_cut = -Env_sum_cut;
    corr_val = -corr_val;
end

time_axis_windows = ((cut_idx-1):N_epochs-1) * Ssize + (Wsize/2);

figure('Name', 'Сравнение огибающих', 'Position', [100, 100, 800, 400]);
plot(time_axis_windows, target_env_downsampled_cut, 'k', 'LineWidth', 2, 'DisplayName', 'Оригинальная огибающая (Ground Truth)');
hold on;
plot(time_axis_windows, Env_sum_cut, 'r--', 'LineWidth', 2, 'DisplayName', sprintf('Восстановленная огибающая (LPP), r = %.2f', corr_val));
title(sprintf('Поиск индуцированной активности (10 субъектов, %d секунд)', Ts));
xlabel('Время (с)');
ylabel('Амплитуда (z-score)');
legend;
grid on;

print('envelope_comparison.png', '-dpng', '-r300');
disp('Готово! График сохранен в envelope_comparison.png');
