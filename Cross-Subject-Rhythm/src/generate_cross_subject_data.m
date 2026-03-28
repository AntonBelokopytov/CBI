function [X_all_subjects, target_env] = generate_cross_subject_data(G, N_subj, N_distr, N_bg, Fs, Ts, flanker, SNR)
% Генерирует межсубъектные данные с индуцированной активностью.
% Вход:
% G       - матрица свинцового поля (Leadfield matrix), размер [Nsens, Nsites*3]
% N_subj  - количество субъектов
% N_distr - количество распределенных целевых источников (обычно 1 или 2)
% N_bg    - количество фоновых локальных источников
% Fs      - частота дискретизации (Гц)
% Ts      - длина сигнала в секундах
% flanker - отступы по краям в секундах для устранения краевых эффектов фильтрации
% SNR     - отношение сигнал/шум для амплитуды целевой огибающей
%
% Выход:
% X_all_subjects - 3D матрица сгенерированных данных [Nsens, Nsamples, N_subj]
% target_env     - общая сгенерированная огибающая целевого ритма [1, Nsamples]

if nargin < 8
    SNR = 1;
end

Gx = G(:,1:3:end);
Gy = G(:,2:3:end);
Gz = G(:,3:3:end);
[Nsens, Nsites] = size(Gx);

N = Ts * Fs;
flanker_samples = flanker * Fs;
total_samples = N + 2*flanker_samples;

% 1. Генерируем ОДНУ общую огибающую (target_env) для всех субъектов
[be, ae] = butter(2, 0.5 / (Fs / 2), 'low'); % Фильтр нижних частот 0.5 Гц для огибающей
tm = filtfilt(be, ae, randn(1, total_samples));
tm = tm(flanker_samples+1 : end-flanker_samples);
tm = (tm - mean(tm)) / std(tm);
tm_snr = tm * SNR;
tm_snr = tm_snr - min(tm_snr) + eps; % Положительная огибающая
target_env = tm_snr;

% Массивы фильтров для шума
[bn, an] = butter(4, [1, 35] / (Fs / 2)); % Для белого шума (сенсорный шум)

% Предварительно выделяем память для всех субъектов
X_all_subjects = zeros(Nsens, N, N_subj);

% 2. Цикл по субъектам
for subj_idx = 1:N_subj

    % a) Генерируем уникальное пространственное распределение targetA для этого субъекта
    targetA = zeros(Nsens, N_distr);
    src_indsA = randperm(Nsites, N_distr);
    for i = 1:N_distr
        src_idx = src_indsA(i);
        r = rand(3,1)*2 - 1;
        r = r / norm(r);
        targetA(:,i) = Gx(:,src_idx)*r(1) + Gy(:,src_idx)*r(2) + Gz(:,src_idx)*r(3);
    end

    % b) Генерируем сигналы источников для этого субъекта
    [XS, X_bg, X_n] = generate_subject_sources(G, N_bg + N_distr, N_distr, Fs, total_samples, flanker_samples, targetA, target_env);

    % c) Суммируем сигнал, фоновый шум и сенсорный шум
    % Добавляем белый шум X_n
    noise_scale = norm(X_bg, 'fro');
    X = XS + X_bg + 0.1 * X_n * noise_scale / norm(X_n, 'fro');

    X_all_subjects(:,:,subj_idx) = X;
end

end

% =========================================================================
% Локальная функция для генерации сигналов одного субъекта
% =========================================================================
function [X_s, X_bg, X_n] = generate_subject_sources(G, Nsrc, Ndistr, Fs, total_samples, flanker_samples, targetA, target_env)

Gx = G(:,1:3:end);
Gy = G(:,2:3:end);
Gz = G(:,3:3:end);
[Nsens, Nsites] = size(Gx);

% Подготавливаем пространственную матрицу для фоновых источников
GA = zeros(Nsens, Nsrc);
GA(:, 1:Ndistr) = targetA;
src_indsA = randperm(Nsites, Nsrc - Ndistr);

for i = 1:(Nsrc - Ndistr)
    src_idx = src_indsA(i);
    r = rand(3,1)*2 - 1;
    r = r / norm(r);
    GA(:, Ndistr + i) = Gx(:,src_idx)*r(1) + Gy(:,src_idx)*r(2) + Gz(:,src_idx)*r(3);
end

% Фильтры для генерации осцилляций
[b, a] = butter(4, [8, 12] / (Fs / 2)); % Целевой ритм (например, альфа-ритм 8-12 Гц)
[bem, aem] = butter(4, 0.5 / (Fs / 2), 'low'); % Фильтр для огибающей фоновых источников
[ben, aen] = butter(5, 5 / (Fs / 2)); % Фильтр для небольших возмущений огибающей

% 1) Несущие сигналы для всех источников
S = filtfilt(b, a, randn(Nsrc, total_samples)')';
S = S(:, flanker_samples+1 : end-flanker_samples);

% 2) Огибающие для фоновых источников (случайные)
M = filtfilt(bem, aem, randn(Nsrc, total_samples)')';
M = M(:, flanker_samples+1 : end-flanker_samples);

% Заменяем огибающую для целевых источников на общую target_env
for k = 1:Ndistr
    M(k, :) = target_env;
end

% Добавляем вариации к фоновым огибающим и нормализуем
for k = Ndistr+1:Nsrc
    m = M(k,:);
    m = (m - mean(m)) / std(m);

    env_n = filtfilt(ben, aen, randn(1, total_samples));
    env_n = env_n(flanker_samples+1 : end-flanker_samples);
    env_n = env_n ./ norm(env_n);
    m = m + 0.1 * norm(m) * env_n;

    M(k,:) = m - min(m) + eps;
end

% 3) Формируем итоговые сигналы источников (несущая * огибающая)
for k = 1:Nsrc
    S(k,:) = (S(k,:) - mean(S(k,:))) / std(S(k,:));

    % Нормируем текущую огибающую несущей (с помощью преобразования Гильберта)
    % чтобы амплитуда несущей была равна 1 до умножения на M
    env = abs(hilbert(S(k,:)')');
    S(k,:) = S(k,:) ./ (env + eps);

    % Умножаем на нашу заданную огибающую
    S(k,:) = S(k,:) .* M(k,:);
    S(k,:) = S(k,:) - mean(S(k,:));
end

% 4) Проецируем источники на сенсоры (через матрицу G)
X_s = GA(:, 1:Ndistr) * S(1:Ndistr, :);
X_bg = GA(:, Ndistr+1:end) * S(Ndistr+1:end, :);

% 5) Генерируем сенсорный белый шум
[bn, an] = butter(4, [1, 35] / (Fs / 2));
X_n = filtfilt(bn, an, randn(Nsens, total_samples)')';
X_n = X_n(:, flanker_samples+1 : end-flanker_samples);
X_n = X_n - mean(X_n, 2);
X_n = X_n ./ std(X_n, 0, 2);

end
