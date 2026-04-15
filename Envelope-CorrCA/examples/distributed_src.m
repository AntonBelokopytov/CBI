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

%% Применение Лапласиана
Patterns = squeeze(A(1,:,:));

% 1. Подготовка данных
X = fwd_model.sensor_pos(:,1);
Y = fwd_model.sensor_pos(:,2);
Z = fwd_model.sensor_pos(:,3);
V = Patterns; 

% Проекция на единичную сферу (обязательно для метода Перрена)
r = sqrt(X.^2 + Y.^2 + Z.^2);
X = X./r; Y = Y./r; Z = Z./r;

n_channels = size(V, 1);
n_patterns = size(V, 2);

% 2. Параметры алгоритма
m = 4;          % Параметр жесткости (обычно 4)
n_leg = 50;     % Количество полиномов Лежандра для аппроксимации
lambda = 1e-5;  % Параметр регуляризации (сглаживание)

% 3. Вычисление матрицы косинусных расстояний
% cos_dist(i,j) - косинус угла между электродом i и j
cos_dist = [X, Y, Z] * [X, Y, Z]';

cos_dist(cos_dist > 1) = 1; 
cos_dist(cos_dist < -1) = -1;

% 4. Расчет матриц G и H (Векторизованный вариант)
G = zeros(n_channels, n_channels);
H = zeros(n_channels, n_channels);

for n = 1:n_leg
    % Вычисляем полином Лежандра степени n для всей матрицы косинусов сразу
    % legendre(n, X) для матрицы выдает 3D массив, нам нужна первая страница (m=0)
    P = legendre(n, cos_dist);
    Pn = squeeze(P(1,:,:)); 
    
    % Множители
    common_denom = (n^m * (n+1)^m);
    multiplier = (2*n + 1) / common_denom;
    
    % Накапливаем результат
    G = G + multiplier * Pn;
    H = H + multiplier * (n * (n+1)) * Pn;
end

G = G / (4*pi);
H = H / (4*pi);

% 5. Решение системы уравнений и нахождение Лапласиана
% Добавляем константу (среднее) для корректного решения
G_reg = G + eye(n_channels) * lambda; 

% Для каждого паттерна находим коэффициенты C
% Мы решаем систему: G*C = V (центрированное)
V_centered = V - mean(V, 1); 
C = G_reg \ V_centered; 

% Вычисляем Лапласиан: L = H * C
Patterns_CSD = H * C;

%%
n_patterns = size(Patterns_CSD, 2);
n_sources = size(fwd_model.A, 2) / 3;
z = zeros(n_patterns, n_sources);

for patt_i = 1:n_patterns
    patt_i
    a_vec = Patterns_CSD(:, patt_i);     
    for s = 1:n_sources
        cols = (s-1)*3 + 1 : s*3;
        U = fwd_model.A(:, cols);        
        alpha = (U \ a_vec);
        Ualpha = U * alpha;
        
        z(patt_i, s) = norm(a_vec - Ualpha);
    end
end

%%
plot(z(1,:))

%%
src = 1; 
[~, best_src] = min(z(src,:)); 

best_coord = source_pos(best_src,:);
a_vec = Patterns(:, src);     

cols = (best_src-1)*3 + 1 : best_src*3;
U = fwd_model.A(:, cols);        
alpha = U \ a_vec; 

figure('Name', 'Best Dipole Fit', 'Color', 'w', 'Position', [100, 100, 800, 600]);
hold on; grid on;

scatter3(source_pos(:,1), source_pos(:,2), source_pos(:,3), ...
    5, [0.7 0.7 0.7], 'filled', 'MarkerFaceAlpha', 0.5);

scatter3(best_coord(1), best_coord(2), best_coord(3), ...
    100, 'r', 'filled');

alpha_norm = alpha / norm(alpha); 
arrow_length = 0.04; 

quiver3(best_coord(1), best_coord(2), best_coord(3), ...
        alpha_norm(1) * arrow_length, ...
        alpha_norm(2) * arrow_length, ...
        alpha_norm(3) * arrow_length, ...
        0, 'Color', 'b', 'LineWidth', 3, 'MaxHeadSize', 2); 

view(3);
axis equal
set(gca, 'Visible', 'off');
title(['Best Dipole Fit for Pattern ', num2str(src)], 'FontSize', 14);
hold off;

%%
src = 4; 
N_best = 100; % Количество топовых решений

% 1. Сортируем все источники по возрастанию ошибки
[sorted_z, sort_idx] = sort(z(src,:), 'ascend');
top_indices = sort_idx(1:N_best);
top_errors = sorted_z(1:N_best);

% 2. Рассчитываем веса для масштабирования
min_err = top_errors(1);
max_err = top_errors(end);

if max_err == min_err
    fit_weights = ones(N_best, 1);
else
    fit_weights = 1 - 0.8 * ((top_errors - min_err) / (max_err - min_err));
    fit_weights = fit_weights(:); % Убеждаемся, что это вектор-столбец
end

% Подготавливаем массивы для координат и векторов alpha
top_coords = zeros(N_best, 3);
top_alphas = zeros(N_best, 3);
a_vec = Patterns(:, src);     

% Проходим по всем выбранным источникам, чтобы собрать данные
for i = 1:N_best
    curr_src = top_indices(i);
    top_coords(i, :) = source_pos(curr_src, :);
    
    cols = (curr_src-1)*3 + 1 : curr_src*3;
    U = fwd_model.A(:, cols);        
    alpha = U \ a_vec; 
    top_alphas(i, :) = alpha'; % Транспонируем в строку
end

% --- ВЫЧИСЛЕНИЕ ЭКВИВАЛЕНТНОГО (СРЕДНЕГО) ДИПОЛЯ ---
% Нормируем веса так, чтобы их сумма была равна 1 для корректного усреднения
norm_weights = fit_weights / sum(fit_weights);

% Взвешенный центр масс (эпицентр)
avg_coord = sum(top_coords .* norm_weights, 1);

% Взвешенное среднее направление диполя
avg_alpha = sum(top_alphas .* norm_weights, 1);
avg_alpha_norm = avg_alpha / norm(avg_alpha); 
arrow_length = 0.05; 

% 3. ВИЗУАЛИЗАЦИЯ
figure('Name', 'Top Dipole Cluster', 'Color', 'w', 'Position', [100, 100, 800, 600]);
hold on; grid on;

% Рисуем кору головного мозга (прозрачность 0.1 для контраста с кластером)
scatter3(source_pos(:,1), source_pos(:,2), source_pos(:,3), ...
    5, [0.7 0.7 0.7], 'filled', 'MarkerFaceAlpha', 0.5);

% Рисуем кластер решений одним вызовом (с градиентом)
% Размеры точек зависят от веса (от 20 до 100)
% Цвет зависит от fit_weights
scatter3(top_coords(:,1), top_coords(:,2), top_coords(:,3), ...
    100 * fit_weights, fit_weights, 'filled', 'MarkerFaceAlpha', 0.8);

% Применяем цветовую карту (например, 'autumn' - от красного к желтому, 
% или 'jet', 'parula')
colormap('parula'); 
cb = colorbar;
cb.Label.String = 'Fitting Weight (1 = Best Fit)';

% Рисуем ОДНУ среднюю стрелку в центре масс
quiver3(avg_coord(1), avg_coord(2), avg_coord(3), ...
        avg_alpha_norm(1) * arrow_length, ...
        avg_alpha_norm(2) * arrow_length, ...
        avg_alpha_norm(3) * arrow_length, ...
        0, 'Color', 'r', 'LineWidth', 4, 'MaxHeadSize', 2); 

view(3);
axis equal;
set(gca, 'Visible', 'off');
title(['Equivalent Dipole for Pattern ', num2str(src)], 'FontSize', 14);
hold off;

%% Визуализация кластеров диполей: Центроид и Среднее направление
gl_s = 1:4;          % Массив с номерами паттернов
N_best = 200;        % Количество топовых решений для каждого паттерна
line_length = 0.05;  % Длина отрезка (диполя) в метрах

% Цвета для разных паттернов
colors = lines(length(gl_s)); 

figure('Name', 'Multi-Source Dipole Clusters (Centroid & Avg Direction)', 'Color', 'w', 'Position', [100, 100, 900, 700]);
hold on; grid on;

% 1. Рисуем кору головного мозга (фоновое облако источников)
scatter3(source_pos(:,1), source_pos(:,2), source_pos(:,3), ...
    5, [0.7 0.7 0.7], 'filled', 'MarkerFaceAlpha', 0.4, 'HandleVisibility', 'off');

h_lines = zeros(1, length(gl_s));
legend_labels = cell(1, length(gl_s));

% 2. Основной цикл по паттернам
for idx = 1:length(gl_s)
    src = gl_s(idx);
    base_color = colors(idx, :);
    
    % --- АНАЛИТИКА ФИТТИНГА ---
    [sorted_z, sort_idx] = sort(z(src,:), 'ascend');
    top_indices = sort_idx(1:N_best);
    top_errors = sorted_z(1:N_best);
    
    % Веса для облака (пропорциональны точности)
    min_err = top_errors(1);
    max_err = top_errors(end);
    if max_err == min_err
        fit_weights = ones(N_best, 1);
    else
        fit_weights = 1 - 0.8 * ((top_errors - min_err) / (max_err - min_err));
    end
    
    top_coords = zeros(N_best, 3);
    top_alphas = zeros(N_best, 3);
    a_vec = Patterns(:, src); 
    
    for i = 1:N_best
        curr_src = top_indices(i);
        top_coords(i, :) = source_pos(curr_src, :);
        
        cols = (curr_src-1)*3 + 1 : curr_src*3;
        U = fwd_model.A(:, cols);        
        alpha = U \ a_vec; 
        top_alphas(i, :) = alpha'; 
    end
    
    % --- ВИЗУАЛИЗАЦИЯ КЛАСТЕРА ---
    gray_color = [0.85 0.85 0.85];
    C = fit_weights(:) * base_color + (1 - fit_weights(:)) * gray_color;
    
    % Рисуем облако "побочных" решений
    scatter3(top_coords(:,1), top_coords(:,2), top_coords(:,3), ...
        80 * fit_weights, C, 'filled', 'MarkerFaceAlpha', 0.5, 'HandleVisibility', 'off');
    
    % --- РАСЧЕТ ЦЕНТРОИДА И СРЕДНЕГО НАПРАВЛЕНИЯ ---
    
    % 1. Центроид — средняя координата всех точек кластера
    centroid_pos = mean(top_coords, 1);
    
    % 2. Среднее направление (вектор ориентации)
    % Важно: если векторы имеют разную полярность (±), среднее может обнулиться.
    % Но в рамках одного кластера паттерна ориентация обычно сонаправлена.
    avg_alpha = mean(top_alphas, 1); 
    avg_alpha_norm = avg_alpha / norm(avg_alpha); % Приводим к единичной длине
    
    % 3. Вычисляем концы отрезка
    p1 = centroid_pos - (line_length / 2) * avg_alpha_norm;
    p2 = centroid_pos + (line_length / 2) * avg_alpha_norm;
    
    % Отрисовка результирующего диполя
    h_lines(idx) = plot3([p1(1) p2(1)], [p1(2) p2(2)], [p1(3) p2(3)], ...
            '-', 'Color', base_color, 'LineWidth', 6); % Чуть жирнее для акцента
        
    % legend_labels{idx} = sprintf('Pattern %d (Centroid/Avg)', src);
end

% Финальные штрихи
view(3);
axis equal;
set(gca, 'Color', 'w', 'XColor', 'none', 'YColor', 'none', 'ZColor', 'none');
% title('Equivalent Dipole Clusters: Centroid & Average Orientation', 'FontSize', 14);

% legend(h_lines, legend_labels, 'Location', 'northeastoutside', 'FontSize', 10);

camlight; 
lighting gouraud;
hold off;

%%
gl_c = 1;
comp_idx = 6;
wx = squeeze(W(gl_c,:,comp_idx))';
patt = squeeze(A(gl_c,:,comp_idx));
% wx = squeeze(W(:,comp_idx));
% patt = squeeze(A(:,comp_idx));
patt = Patterns_CSD(:,comp_idx);

patt = patt * sign(patt(abs(patt)==max(abs(patt))));

clear Yenv Yseg
for ep_idx=1:size(Epochs,3)
    ep = Epochs(:,:,ep_idx);
    ep = ep / sqrt(trace(cov(ep)));
    en = abs(hilbert(ep*wx));
    Yenv(:,ep_idx) = en;
    Yseg(:,ep_idx) = Epochs(:,:,ep_idx)*wx;
end

Filt = wx;

elec = Epochs_inf.hdr.elec
elec.chanpos  = elec.chanpos(1:38, :);
elec.elecpos  = elec.elecpos(1:38, :);
elec.chantype = elec.chantype(1:38);
elec.chanunit = elec.chanunit(1:38);
elec.label    = elec.label(1:38);

figure; hold on; grid on
set(gcf,'Color','w');

tsec     = linspace(-3, 4, size(Yenv,1));
E        = size(Yenv, 2);
env_mean = mean(Yenv, 2, 'omitnan');
sd       = std (Yenv, 0, 2, 'omitnan');

% Подготовка FieldTrip layout
lay = ft_prepare_layout(struct('elec', elec));

% ==== ГЛАВНАЯ РАСКЛАДКА ====================================================
t = tiledlayout(3,2, 'TileSpacing','compact', 'Padding','compact');

% ---- (Left) Heatmap: все эпохи, огибающая ----
axH = nexttile(t, 1, [3 1]);          % левая колонка, 3 строки
imagesc(axH, tsec, 1:E, Yenv');
set(axH,'YDir','normal','Color','w'); grid(axH,'on');
xline(axH, 0, 'k--', 'LineWidth', 2);
% xline(axH, -2, 'k--', 'LineWidth', 2);
xlabel(axH,'time, s'); ylabel(axH,'epoch');
title(axH,'Envelope per epoch');
colorbar(axH);
caxis(axH, [0 3*max(std(Yenv))]);

% ---- (Right-Top) ERP без усреднения ----
axERP = nexttile(t, 2); hold(axERP,'on'); grid(axERP,'on');
set(axERP,'Color','w');
plot(axERP, tsec, Yseg);
xline(axERP, 0, 'k--', 'LineWidth', 2);
% xline(axERP, -2, 'k--', 'LineWidth', 2);
xlabel(axERP,'time, s'); ylabel(axERP,'amplitude');
title(axERP,'Source activity (all epochs)');

% ---- (Right-Middle) Средняя огибающая ± SD ----
axENV = nexttile(t, 4); hold(axENV,'on'); grid(axENV,'on');
set(axENV,'Color','w');
xfill = [tsec, fliplr(tsec)];
yfill = [ (env_mean+sd).', fliplr((env_mean-sd).') ];
fill(axENV, xfill, yfill, [0.3 0.5 1.0], 'FaceAlpha',0.2, 'EdgeColor','none');
plot(axENV, tsec, env_mean, 'Color', [0.1 0.3 0.9], 'LineWidth', 2);
xline(axENV, 0, 'k--', 'LineWidth', 2);
% xline(axENV, -2, 'k--', 'LineWidth', 2);
xlabel(axENV,'time, s'); ylabel(axENV,'envelope (a.u.)');
title(axENV,'Mean envelope \pm SD');

% ==== (Right-Bottom) ДВА ГРАФИКА: ФИЛЬТР и ПАТТЕРН =========================

% 1) Создаём временную ось в тайле #6, берём её позицию и удаляем
axTmp = nexttile(t, 6);
pos6  = axTmp.OuterPosition;   % можно взять Position, если так удобнее
delete(axTmp);

% 2) На это место ставим панель с белым фоном и вложенный tiledlayout 1×2
figCol = get(gcf,'Color');     % обычно 'w'
p6 = uipanel('Parent', gcf, ...
             'Units','normalized', ...
             'Position', pos6, ...
             'BorderType','none', ...
             'BackgroundColor', figCol);   % белый фон панели

t6 = tiledlayout(p6, 1, 2, 'TileSpacing','compact', 'Padding','compact');

% --------- Левый подтайл: ТОПОГРАФИЯ ФИЛЬТРА ---------
axFiltTopo = nexttile(t6, 1);
set(axFiltTopo, 'Color','w');

topoF = [];
topoF.dimord = 'chan_time';
topoF.label  = elec.label;
topoF.time   = 0;
topoF.avg    = Filt;
topoF.elec   = elec;

cfg = [];
cfg.figure       = axFiltTopo;
cfg.layout       = lay;
cfg.comment      = 'no';
cfg.style        = 'fill';
cfg.markersymbol = 'o';
cfg.zlim         = 'maxmin';

cfg.layout.pos(:, 1:2) = cfg.layout.pos(:, 1:2) * 1.1; 
cfg.layout.pos(:, 2) = cfg.layout.pos(:, 2) - 0.05;

% cfg.colorbar     = 'EastOutside';
ft_topoplotER(cfg, topoF);
title(axFiltTopo,'Filter');

% --------- Правый подтайл: ТОПОГРАФИЯ ПАТТЕРНА ---------
axPatTopo = nexttile(t6, 2);
set(axPatTopo, 'Color','w');

valsPat = patt; valsPat = valsPat(:);
topoP = [];
topoP.dimord = 'chan_time';
topoP.label  = elec.label;
topoP.time   = 0;
topoP.avg    = valsPat;
topoP.elec   = elec;

cfg = [];
cfg.figure       = axPatTopo;
cfg.layout       = lay;
cfg.comment      = 'no';
cfg.style        = 'fill';
cfg.markersymbol = 'o';
cfg.zlim         = 'maxmin';

cfg.layout.pos(:, 1:2) = cfg.layout.pos(:, 1:2) * 1.1; 
cfg.layout.pos(:, 2) = cfg.layout.pos(:, 2) - 0.05;

% cfg.colorbar     = 'EastOutside';
ft_topoplotER(cfg, topoP); 
title(axPatTopo,'Pattern');

% На всякий случай перекрасим все оси внутри панели в белый
set(findall(p6, 'type','axes'), 'Color', figCol);

% ==== Синхронизация осей по X у временных графиков =========================
linkaxes([axH axERP axENV], 'x');

% Жестко фиксируем границы от -3 до 4
xlim(axH, [-3, 4]);
