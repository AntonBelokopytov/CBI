close all
clear
clc

poolobj = gcp('nocreate');
if isempty(poolobj)
    parpool;
end

ft_path = 'C:\Users\anton\Documents\GitHub\CBI\site-packages\fieldtrip\';
if ~exist('ft_defaults','file')
    addpath(ft_path);
end

%% 1. Загрузка данных
sub_path = 'D:\OS(CURRENT)\data\smile_lobe\cleaned_data\E001_ptp150_ica_interpolated_raw.fif';
cfg = [];
cfg.dataset = sub_path; 
Epochs_inf = ft_preprocessing(cfg); 
Fs = Epochs_inf.hdr.Fs;

X = Epochs_inf.trial{1};
n_channels = size(X, 1);

%% 2. Настройки целевой частоты и эпохирование
Fc = 30;

% Задаем динамическую полосу (+/- 20% от Fc, но не уже +/- 2 Гц)
band_halfwidth = max(2, Fc * 0.20);
Fmin = Fc - band_halfwidth;
Fmax = Fc + band_halfwidth;
band = [Fmin, Fmax];

% Фильтрация
[b_band, a_band] = butter(2, band/(Fs/2));
X_filt = filtfilt(b_band, a_band, X')';

% Эпохирование (ширина окна привязана строго к центральной частоте)
Ws = 1/Fc; 
Ss = Ws; % Сдвиг окна (без перекрытия)
X_epochs = epoch_data(X_filt', Fs, Ws, Ss);

n_epochs = size(X_epochs, 3);
fprintf('Выбрана частота %d Гц: окно %.3f с, количество эпох: %d\n', Fc, Ws, n_epochs);

%% 3. Вычисление ковариаций с регуляризацией
lambda = 0.05; 
I = eye(n_channels); 
X_covs = zeros(n_channels, n_channels, n_epochs);

for i = 1:n_epochs
    C_sample = cov(X_epochs(:,:,i));    
    nu = trace(C_sample) / n_channels;    
    C_reg = (1 - lambda) * C_sample + lambda * nu * I;    
    X_covs(:,:,i) = C_reg;
end

%% 4. Оптимизированное вычисление матрицы Римановых расстояний
Dists = zeros(n_epochs, n_epochs); 

parfor i = 1:n_epochs-1
    fprintf('%d\n', i);
    row_dists = zeros(1, n_epochs);
    C1 = X_covs(:,:,i);        
    C1_inv = inv(C1); 
    
    for j = i+1:n_epochs
        C2 = X_covs(:,:,j);
        eigs_val = eig(C1_inv * C2);            
        row_dists(j) = sqrt(sum(log(eigs_val).^2));
    end
    Dists(i, :) = row_dists;     
end

Dists = Dists + Dists';
Dists(logical(eye(n_epochs))) = 0; 

%% 5. UMAP проекция
fprintf('Считаем UMAP для %d Гц...\n', Fc);
u = UMAP('n_neighbors', 20, 'n_components', 2, 'metric', 'precomputed');
R = u.fit_transform(Dists); 
fprintf('--- UMAP Готов ---\n');

%% 6. Интерактивный выбор кластеров
figure('Name', sprintf('Риманово многообразие (Fc = %d Гц)', Fc), 'Color', 'w', 'Position', [200, 150, 800, 600]);

% Отрисовка исходного скаттера (серый полупрозрачный фон)
scatter(R(:,1), R(:,2), 20, 'filled', 'MarkerFaceAlpha', 0.4, 'MarkerFaceColor', [0.6 0.6 0.6]);
xlabel('UMAP 1', 'FontSize', 12);
ylabel('UMAP 2', 'FontSize', 12);
axis square;
grid on;
hold on; 

% Инициализация
selected_clusters_idx = {}; 
cluster_count = 0;
color_palette = lines(10); 

disp('---------------------------------------------------');
disp('ИНТЕРАКТИВНЫЙ РЕЖИМ АКТИВИРОВАН:');
disp('Вы можете последовательно выделить любое количество кластеров.');
disp('Двойной клик для завершения выделения текущего кластера.');
disp('---------------------------------------------------');

keep_selecting = true;
while keep_selecting
    cluster_count = cluster_count + 1;
    current_color = color_palette(mod(cluster_count-1, 10) + 1, :); 
    
    title(sprintf('Выделение КЛАСТЕРА %d\nОбведите точки (Двойной клик для завершения)', cluster_count), 'FontSize', 12, 'Color', current_color);
    
    roi = drawpolygon('Color', current_color, 'LineWidth', 1.5, 'FaceAlpha', 0.1);
    
    if ~isvalid(roi)
        fprintf('Окно закрыто. Прерывание выбора.\n');
        break;
    end
    
    poly_pos = roi.Position;
    in_cluster_mask = inpolygon(R(:,1), R(:,2), poly_pos(:,1), poly_pos(:,2));
    current_idx = find(in_cluster_mask);
    
    selected_clusters_idx{cluster_count} = current_idx;
    
    % Перекрашиваем выбранные точки
    scatter(R(current_idx, 1), R(current_idx, 2), 35, current_color, 'filled', 'MarkerEdgeColor', 'k');
    delete(roi);
    
    fprintf('Кластер %d успешно выделен (Эпох: %d).\n', cluster_count, length(current_idx));
    
    answer = questdlg('Хотите выделить еще один кластер?', ...
                      'Интерактивный выбор', ...
                      'Да', 'Нет (Завершить)', 'Да');
                  
    if strcmp(answer, 'Нет (Завершить)') || isempty(answer)
        keep_selecting = false;
    end
end

title(sprintf('Выбор завершен. Сохранено кластеров: %d', length(selected_clusters_idx)), 'FontSize', 13, 'Color', 'k');
hold off;

%% 7. Сопоставление с экспериментальными условиями
T = readtable('D:\OS(CURRENT)\data\smile_lobe\cleaned_data\E001_segment_manifest.csv');
num_clusters = length(selected_clusters_idx);
cluster_stats = struct();

for c = 1:num_clusters
    ids = selected_clusters_idx{c};
    t_starts = (ids - 1) * Ss; 
    
    conditions_found = cell(size(t_starts));
    
    for i = 1:length(t_starts)
        t = t_starts(i);
        row_idx = find(t >= T.concat_t0 & t < T.concat_t1, 1);
        
        if ~isempty(row_idx)
            conditions_found{i} = T.condition{row_idx};
        else
            conditions_found{i} = 'unknown';
        end
    end
    
    unique_conds = unique(T.condition);
    counts = zeros(1, length(unique_conds));
    for k = 1:length(unique_conds)
        counts(k) = sum(strcmp(conditions_found, unique_conds{k}));
    end
    
    cluster_stats(c).cluster_name = sprintf('Cluster %d', c);
    cluster_stats(c).conditions = unique_conds;
    cluster_stats(c).counts = counts;
    cluster_stats(c).ratios = counts / sum(counts) * 100;
end

%% 8. Визуализация распределения для ВСЕХ выделенных кластеров
if num_clusters > 0
    figure('Name', 'Распределение кластеров по условиям', 'Color', 'w', 'Position', [300, 200, 1000, 600]);
    cols = ceil(sqrt(num_clusters));
    rows = ceil(num_clusters / cols);
    
    % Задаем палитру для столбцов, чтобы избежать ошибок с индексами
    bar_colors = lines(length(unique_conds)); 
    
    for c = 1:num_clusters
        subplot(rows, cols, c);
        
        b = bar(cluster_stats(c).ratios, 'FaceColor', 'flat');
        
        % Безопасная раскраска столбцов
        for k = 1:length(cluster_stats(c).ratios)
            b.CData(k,:) = bar_colors(k,:);
        end
        
        set(gca, 'XTickLabel', cluster_stats(c).conditions, 'FontSize', 10);
        ylabel('Процент эпох (%)', 'FontSize', 10);
        title(sprintf('Кластер %d (n = %d)', c, sum(cluster_stats(c).counts)), 'FontSize', 12);
        
        ylim([0 100]); 
        grid on;
    end
end