close all
clear
clc

ft_path = 'C:\Users\ansbel\Documents\GitHub\CBI\site-packages\fieldtrip\';

if ~exist('ft_defaults','file')
    addpath(ft_path);
end

%%
fwd_model = load('fsaverage_38ch_leadfield.mat');

%% Target epochs
sub_path = 'D:\OS(CURRENT)\data\parkinson\pathology\Patient_1_CenterOut_OFF_EEG_clean_epochs.fif';
% sub_path = 'D:\OS(CURRENT)\data\parkinson\control\Control_3_CenterOut_epochs.fif';
cfg = [];
cfg.dataset = sub_path; 
Epochs_inf = ft_preprocessing(cfg); 

Fs = Epochs_inf.hdr.Fs;

[~, n_ts_ep] = size(Epochs_inf.trial{1});

%%
idxs = [  0,   1,   2,   3,   6,   8,  12,  13,  15,  16,  17,  19,  22,...
        24,  25,  28,  30,  32,  33,  35,  37,  39,  41,  42,  44,  45,...
        48,  50,  51,  55,  57,  58,  59,  62,  63,  65,  69,  70,  73,...
        74,  75,  78,  79,  82,  85,  88,  89,  92,  93,  94,  97, 100,...
       101, 103, 105, 106, 110, 111, 112, 117, 118, 121, 122, 124, 126,...
       128, 131, 132, 134, 136, 137, 138] + 1;
% idxs = [  1,   3,   4,   5,   7,   8,  12,  14,  15,  17,  21,  22,  25,...
%         26,  28,  29,  31,  33,  36,  37,  39,  40,  42,  44,  48,  49,...
%         50,  53,  55,  56,  59,  61,  63,  64,  66,  67,  71,  72,  73,...
%         74,  76,  77,  80,  81,  85,  86,  89,  90,  92,  94,  98,  99,...
%        101, 103, 108, 109, 110, 111, 114, 116, 117, 120, 121, 124, 125,...
%        127, 128, 130, 131, 134, 136, 138, 141, 142, 144, 146, 147] + 1;

all_idxd = 1:numel(Epochs_inf.trial);
% idxs = setdiff(all_idxd, idxs); 

Fc = 12;
band_halfwidth = max(2, Fc * 0.10);

Fmin = Fc - band_halfwidth;
Fmax = Fc + band_halfwidth;
band = [Fmin Fmax];

Wsize = 1/Fc;
Ssize = Wsize/2;

[b_band,a_band] = butter(4, band/(Fs/2));

Fs = Epochs_inf.hdr.Fs;

clear Epochs Epochs_alg
for ep_idx=1:numel(Epochs_inf.trial)  
    Ep = Epochs_inf.trial{ep_idx}';
    Ep = Ep(:,1:38);
    Epfilt = filtfilt(b_band,a_band,Ep);
    Epochs(:,:,ep_idx) = Epfilt;
    Epochs_alg(:,:,ep_idx) = Epfilt(Fs/2+1:end-Fs/2,:);
end

Epochs = Epochs(:,:,idxs);
Epochs_alg = Epochs_alg(:,:,idxs);

[W, A, z_trials, X_epochs] = env_corrca(Epochs_alg, Fs, Wsize, Ssize);

%%
figure
plot(mean(z_trials(:,1,:),3))

%%
source_pos = fwd_model.source_pos; 
sensor_pos = fwd_model.sensor_pos; 
figure('Name', 'MNE Forward Model 3D', 'Color', 'w');

scatter3(source_pos(:,1), source_pos(:,2), source_pos(:,3), ...
    5, [0.7 0.7 0.7], 'filled', 'MarkerFaceAlpha', 0.1);

hold on

scatter3(sensor_pos(:,1), sensor_pos(:,2), sensor_pos(:,3), ...
    60, 'r', 'filled', 'MarkerEdgeColor', 'k');

%%
Patterns = squeeze(A(1,:,:));

%%
n_patterns = size(Patterns, 2);
n_sources = size(fwd_model.A, 2);
z = zeros(n_patterns, n_sources);

% Предварительно нормируем матрицу лидфилда (столбцы), чтобы не делать это в цикле
% Это значительно ускорит процесс
A_normed = fwd_model.A ./ sqrt(sum(fwd_model.A.^2, 1));

for patt_i = 1:n_patterns
    fprintf('Обработка паттерна %d/%d...\n', patt_i, n_patterns);
    
    % Нормируем паттерн ОДИН раз
    a_vec = Patterns(:, patt_i);
    a_vec = a_vec / norm(a_vec);
    
    % Векторизованный расчет невязки (без внутреннего цикла по s)
    % Косинусное расстояние: 1 - |проекция|
    % Использование abs позволяет находить источник независимо от знака паттерна
    z(patt_i, :) = 1 - abs(a_vec' * A_normed);
end

%%
plot(z(1,:))

%% Визуализация кластеров: Фиксированные нормали (Fixed Orientation)
gl_s = 1:4; 
N_best = 100;        
line_length = 0.03;  

colors = lines(length(gl_s)); 
figure('Name', 'Dipole Clusters (Fixed Normals)', 'Color', 'w', 'Position', [100, 100, 900, 700]);
hold on; grid on;

% 1. Рисуем кору
scatter3(source_pos(:,1), source_pos(:,2), source_pos(:,3), ...
    5, [0.7 0.7 0.7], 'filled', 'MarkerFaceAlpha', 0.2, 'HandleVisibility', 'off');

h_lines = zeros(1, length(gl_s));
legend_labels = cell(1, length(gl_s));

% Проверка: есть ли нормали в модели? 
% Если поля source_ori нет, попробуйте найти его в fwd_model.label или других полях
if ~isfield(fwd_model, 'source_ori')
    error('В fwd_model не найдено поле source_ori (векторы нормалей). Проверьте структуру модели.');
end

for idx = 1:length(gl_s)
    src = gl_s(idx);
    base_color = colors(idx, :);
    
    [sorted_z, sort_idx] = sort(z(src,:), 'ascend');
    top_indices = sort_idx(1:N_best);
    top_errors = sorted_z(1:N_best);
    
    % Веса для облака
    min_err = top_errors(1);
    max_err = top_errors(end);
    fit_weights = 1 - 0.8 * ((top_errors - min_err) / (max_err - min_err + eps));
    
    % Инициализация (координаты 3D, ориентации 3D, амплитуды 1D)
    top_coords = zeros(N_best, 3); 
    top_oris = zeros(N_best, 3);
    top_amps = zeros(N_best, 1);
    
    a_vec = Patterns(:, src); 
    
    for i = 1:N_best
        curr_src = top_indices(i);
        
        % 1. Запоминаем 3D координату
        top_coords(i, :) = source_pos(curr_src, :);
        
        % 2. Берем готовую нормаль из модели для этой точки
        ori = fwd_model.source_ori(curr_src, :); 
        
        % 3. Оцениваем амплитуду (скаляр), чтобы понять полярность (внутрь/наружу)
        U = fwd_model.A(:, curr_src); % Это вектор усиления для нормали
        amp = U \ a_vec; % Проекция паттерна на Leadfield
        
        top_amps(i) = amp;
        % Умножаем нормаль на знак амплитуды, чтобы диполь смотрел в нужную сторону
        top_oris(i, :) = ori * sign(amp); 
    end
    
    % --- ВИЗУАЛИЗАЦИЯ КЛАСТЕРА ---
    gray_color = [0.85 0.85 0.85];
    C = fit_weights(:) * base_color + (1 - fit_weights(:)) * gray_color;
    scatter3(top_coords(:,1), top_coords(:,2), top_coords(:,3), ...
        80 * fit_weights, C, 'filled', 'MarkerFaceAlpha', 0.4, 'HandleVisibility', 'off');
    
    % --- РАСЧЕТ ЦЕНТРОИДА И СРЕДНЕЙ НОРМАЛИ ---
    
    % Геометрический центр
    centroid_pos = mean(top_coords, 1);
    
    % Средний вектор нормали в этом кластере
    avg_ori = mean(top_oris, 1); 
    avg_ori_norm = avg_ori / norm(avg_ori);
    
    % Вычисляем концы отрезка
    p1 = centroid_pos - (line_length / 2) * avg_ori_norm;
    p2 = centroid_pos + (line_length / 2) * avg_ori_norm;
    
    % Рисуем результирующий диполь
    h_lines(idx) = plot3([p1(1) p2(1)], [p1(2) p2(2)], [p1(3) p2(3)], ...
            '-', 'Color', base_color, 'LineWidth', 6);
        
    legend_labels{idx} = sprintf('Pattern %d (Fixed Normal)', src);
end

view(3); axis equal;
title('Source Localization with Normal Constraints', 'FontSize', 14);
legend(h_lines, legend_labels, 'Location', 'northeastoutside');
hold off;









