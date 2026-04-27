close all
clear
clc

ft_path = 'C:\Users\anton\Documents\GitHub\CBI\site-packages\fieldtrip';

if ~exist('ft_defaults','file')
    addpath(ft_path);
end

ft_defaults;

%% =====================================================================
% LOAD DATA
% =====================================================================
sub_path = 'D:\OS(CURRENT)\data\parkinson\pathology\Patient_1_CenterOut_OFF_EEG_clean_epochs.fif';
% sub_path = 'D:\OS(CURRENT)\data\parkinson\control\Control_3_CenterOut_epochs.fif';

cfg = [];
cfg.dataset = sub_path;
Xinf = ft_preprocessing(cfg);    % Load EEG/MEG data
Fs = Xinf.fsample;               % Sampling frequency

% Initialize topography structure
topo = [];
topo.dimord = 'chan_time';
topo.label  = Xinf.elec.label;  
topo.time   = 0;
topo.elec   = Xinf.elec;
topo.time    = 0;

% Prepare FieldTrip layout for topography plotting
laycfg = [];
laycfg.elec = Xinf.elec;
lay = ft_prepare_layout(laycfg);     

cfg.marker       = 'labels';
cfg.layout       = lay;
cfg.comment      = 'no';
cfg.style        = 'fill';
cfg.markersymbol = 'o';
cfg.colorbar     = 'yes'; 

N_trials = numel(Xinf.trial)

%%
fmin = 9;
fmax = 14;
% Ws = 1/fmin;   
% Ss = Ws/2; 
Ws = 1;   
Ss = 1; 

[b_band, a_band] = butter(4, [fmin fmax] / (Fs/2), 'bandpass');

X_filt = [];
X_tr_epochs = []; 
Z_tr_epochs = [];
for tr_i = 1:N_trials
    tr_i
    X_tr = Xinf.trial{tr_i};
    X_tr = X_tr(1:38,:) - mean(X_tr(1:38,:),2);
    X_tr_filt = filtfilt(b_band, a_band, X_tr'); 
    X_filt(:,:,tr_i) = X_tr_filt;
    temp_epochs = epoch_data(X_tr_filt, Fs, Ws, Ss);
    X_tr_epochs(:,:,:,tr_i) = temp_epochs;

    Z_tr = Xinf.trial{tr_i}(39:end,:);
    temp_epochs = epoch_data(Z_tr', Fs, Ws, Ss);
    temp_epochs = squeeze(mean(temp_epochs.^2,1));
    Z_tr_epochs(:,:,tr_i) = temp_epochs;
end

%%
idxs = [  0,   1,   2,   3,   6,   8,  12,  13,  15,  16,  17,  19,  22,...
        24,  25,  28,  30,  32,  33,  35,  37,  39,  41,  42,  44,  45,...
        48,  50,  51,  55,  57,  58,  59,  62,  63,  65,  69,  70,  73,...
        74,  75,  78,  79,  82,  85,  88,  89,  92,  93,  94,  97, 100,...
       101, 103, 105, 106, 110, 111, 112, 117, 118, 121, 122, 124, 126,...
       128, 131, 132, 134, 136, 137, 138] + 1;

X_eps_cond = X_tr_epochs(:,:,:,idxs);
X_eps = X_eps_cond(:,:,:);

z_eps_cond = Z_tr_epochs(:,:,idxs);
z_eps = z_eps_cond(6:8,:);

X_filt_cond = X_filt(:,:,idxs);

ex_variable = mean(z_eps,1);

%% =====================================================================
% БАЗОВЫЙ (ИДЕАЛЬНЫЙ) РАСЧЕТ eSPoC И TRAIN/TEST SPLIT
% =====================================================================
N_epochs = size(X_eps, 3);

% Считаем ковариации для всех эпох заранее
Epochs_cov = zeros(size(X_eps, 2), size(X_eps, 2), N_epochs);
for ep_idx = 1:N_epochs
    Epochs_cov(:,:,ep_idx) = cov(X_eps(:,:,ep_idx));
end

% 1. Находим Ground Truth на полных данных
[W_true,A_true] = espoc(X_eps, ex_variable);

% Находим "истинную" компоненту (берем последнюю - полюс дезактивации)
w_true_best = W_true(:, end);
a_true_best = A_true(:, end);

% Считаем истинную логарифмическую огибающую
True_envelope = zeros(1, N_epochs);
for ep_i = 1:N_epochs
   True_envelope(ep_i) = w_true_best' * Epochs_cov(:,:,ep_i) * w_true_best;     
end
True_envelope = (True_envelope - mean(True_envelope)) / std(True_envelope);

% 2. ХРОНОЛОГИЧЕСКОЕ разделение на Train и Test
N_train = floor(N_epochs * 0.2);

train_idx = 1:N_train;               % Строго первая половина сессии
test_idx  = (N_train + 1):N_epochs;  % Строго вторая половина сессии
N_test    = length(test_idx);

% Разделяем тензоры данных
X_eps_train = X_eps(:,:,train_idx);
X_eps_test  = X_eps(:,:,test_idx);

Epochs_cov_train = Epochs_cov(:,:,train_idx);
Epochs_cov_test  = Epochs_cov(:,:,test_idx);

% Выделяем и стандартизируем Ground Truth таргеты строго по своим выборкам
True_env_train = True_envelope(train_idx);
True_env_train = (True_env_train - mean(True_env_train)) / std(True_env_train);

True_env_test  = True_envelope(test_idx);
True_env_test  = (True_env_test - mean(True_env_test)) / std(True_env_test);

%% =====================================================================
% ЭКСПЕРИМЕНТ С ЗАГРЯЗНЕНИЕМ ВНЕШНЕЙ ПЕРЕМЕННОЙ (TRAIN)
% =====================================================================
corr_list = 1:-0.05:0.1;  
n_iters = 50;

% 4 метрики для eSPoC
env_train_dirty_espoc = zeros(length(corr_list), n_iters);
env_train_clean_espoc = zeros(length(corr_list), n_iters);
env_test_clean_espoc  = zeros(length(corr_list), n_iters);
pat_corr_espoc        = zeros(length(corr_list), n_iters);

% 4 метрики для SPoC
env_train_dirty_spoc  = zeros(length(corr_list), n_iters);
env_train_clean_spoc  = zeros(length(corr_list), n_iters);
env_test_clean_spoc   = zeros(length(corr_list), n_iters);
pat_corr_spoc         = zeros(length(corr_list), n_iters);

fprintf('Запуск кросс-валидационного эксперимента с шумом...\n');
for c_i = 1:length(corr_list)
    target_r = corr_list(c_i);
    fprintf('Уровень корреляции r = %.2f...\n', target_r);
    
    % Временные переменные для безопасной работы внутри parfor
    tmp_e_dirty = zeros(1, n_iters);
    tmp_e_clean = zeros(1, n_iters);
    tmp_e_test  = zeros(1, n_iters);
    tmp_e_pat   = zeros(1, n_iters);
    
    tmp_s_dirty = zeros(1, n_iters);
    tmp_s_clean = zeros(1, n_iters);
    tmp_s_test  = zeros(1, n_iters);
    tmp_s_pat   = zeros(1, n_iters);
    
    parfor iter = 1:n_iters
        % iter
        % 1. Генерируем ШУМ СТРОГО ОРТОГОНАЛЬНЫЙ к True_env_train
        noise = randn(1, N_train);
        noise = noise - mean(noise);
        
        % Проекция Грама-Шмидта (используем dot() вместо умножения транспонированных векторов)
        proj_coef = dot(noise, True_env_train) / dot(True_env_train, True_env_train);
        noise = noise - proj_coef * True_env_train;
        noise = noise / std(noise); % Нормализуем ортогональный остаток
        
        % Формула смешивания
        z_train_noisy = target_r * True_env_train + sqrt(1 - target_r^2) * noise;
        z_train_noisy = (z_train_noisy - mean(z_train_noisy)) / std(z_train_noisy);
        
        % 2. Обучаем алгоритмы на ГРЯЗНОМ TRAIN
        [We, Ae] = espoct_csp(X_eps_train, z_train_noisy);
        [Ws, As] = spoc(X_eps_train, z_train_noisy); 
        
        % Берем компоненту с макс. положительной корреляцией
        w_e = We(:, 1); a_e = Ae(:, 1);
        w_s = Ws(:, 1); a_s = As(:, 1);
        
        % --- Извлекаем огибающие на TRAIN ---
        env_e_tr = zeros(1, N_train);
        env_s_tr = zeros(1, N_train);
        for ep = 1:N_train
            env_e_tr(ep) = w_e' * Epochs_cov_train(:,:,ep) * w_e;
            env_s_tr(ep) = w_s' * Epochs_cov_train(:,:,ep) * w_s;
        end
        
        % --- Извлекаем огибающие на TEST ---
        env_e_te = zeros(1, N_test);
        env_s_te = zeros(1, N_test);
        for ep = 1:N_test
            env_e_te(ep) = w_e' * Epochs_cov_test(:,:,ep) * w_e;
            env_s_te(ep) = w_s' * Epochs_cov_test(:,:,ep) * w_s;
        end
        
        % --- ЗАПИСЬ 4 МЕТРИК eSPoC во временные массивы ---
        tmp_e_dirty(iter) = abs(corr(env_e_tr', z_train_noisy'));
        tmp_e_clean(iter) = abs(corr(env_e_tr', True_env_train'));
        tmp_e_test(iter)  = abs(corr(env_e_te', True_env_test'));
        tmp_e_pat(iter)   = abs(corr(a_e, a_true_best));
        
        % --- ЗАПИСЬ 4 МЕТРИК SPoC во временные массивы ---
        tmp_s_dirty(iter) = abs(corr(env_s_tr', z_train_noisy'));
        tmp_s_clean(iter) = abs(corr(env_s_tr', True_env_train'));
        tmp_s_test(iter)  = abs(corr(env_s_te', True_env_test'));
        tmp_s_pat(iter)   = abs(corr(a_s, a_true_best));
    end
    
    % Переносим данные из временных массивов в основные (абсолютно безопасно для MATLAB)
    env_train_dirty_espoc(c_i, :) = tmp_e_dirty;
    env_train_clean_espoc(c_i, :) = tmp_e_clean;
    env_test_clean_espoc(c_i, :)  = tmp_e_test;
    pat_corr_espoc(c_i, :)        = tmp_e_pat;
    
    env_train_dirty_spoc(c_i, :)  = tmp_s_dirty;
    env_train_clean_spoc(c_i, :)  = tmp_s_clean;
    env_test_clean_spoc(c_i, :)   = tmp_s_test;
    pat_corr_spoc(c_i, :)         = tmp_s_pat;
end

%% =====================================================================
% ОТРИСОВКА РЕЗУЛЬТАТОВ (4 ГРАФИКА)
% =====================================================================
figure('Position', [50, 50, 1400, 800], 'Color', 'w');
t = tiledlayout(2, 2, 'TileSpacing', 'compact', 'Padding', 'compact');

% 1. Train Dirty
nexttile(t); hold on; grid on;
errorbar(corr_list, mean(env_train_dirty_espoc, 2),   std(env_train_dirty_espoc, 0, 2), ...
    '-o', 'LineWidth', 2, 'DisplayName', 'Riemannian eSPoC');
errorbar(corr_list, mean(env_train_dirty_spoc, 2), std(env_train_dirty_spoc, 0, 2), ...
    '-x', 'LineWidth', 2, 'DisplayName', 'Classical SPoC');
set(gca, 'XDir', 'reverse'); ylim([0 1.05]); xlim([0.05 1.05]);
xlabel('Input Target Quality (r)'); ylabel('Correlation');
title('1. Correlation with DIRTY Train Target', 'FontSize', 12);
legend('Location', 'southwest');

% 2. Train Clean
nexttile(t); hold on; grid on;
errorbar(corr_list, mean(env_train_clean_espoc, 2), std(env_train_clean_espoc, 0, 2), ...
    '-o', 'LineWidth', 2, 'DisplayName', 'Riemannian eSPoC');
errorbar(corr_list, mean(env_train_clean_spoc, 2), std(env_train_clean_spoc, 0, 2), ...
    '-x', 'LineWidth', 2, 'DisplayName', 'Classical SPoC');
set(gca, 'XDir', 'reverse'); ylim([0 1.05]); xlim([0.05 1.05]);
xlabel('Input Target Quality (r)'); ylabel('Correlation');
title('2. Correlation with CLEAN Train Envelope', 'FontSize', 12);
legend('Location', 'southwest');

% 3. Test Clean
nexttile(t); hold on; grid on;
errorbar(corr_list, mean(env_test_clean_espoc, 2), std(env_test_clean_espoc, 0, 2), ...
    '-o', 'LineWidth', 2, 'DisplayName', 'Riemannian eSPoC');
errorbar(corr_list, mean(env_test_clean_spoc, 2), std(env_test_clean_spoc, 0, 2), ...
    '-x', 'LineWidth', 2, 'DisplayName', 'Classical SPoC');
set(gca, 'XDir', 'reverse'); ylim([0 1.05]); xlim([0.05 1.05]);
xlabel('Input Target Quality (r)'); ylabel('Correlation');
title('3. TEST SET: Correlation with CLEAN Envelope', 'FontSize', 12);
legend('Location', 'southwest');

% 4. Pattern
nexttile(t); hold on; grid on;
errorbar(corr_list, mean(pat_corr_espoc, 2), std(pat_corr_espoc, 0, 2), ...
    '-o', 'LineWidth', 2, 'DisplayName', 'Riemannian eSPoC');
errorbar(corr_list, mean(pat_corr_spoc, 2), std(pat_corr_spoc, 0, 2), ...
    '-x', 'LineWidth', 2, 'DisplayName', 'Classical SPoC');
set(gca, 'XDir', 'reverse'); ylim([0 1.05]); xlim([0.05 1.05]);
xlabel('Input Target Quality (r)'); ylabel('Correlation');
title('4. Spatial Pattern Correlation (Accuracy)', 'FontSize', 12);
legend('Location', 'southwest');

%%
src_idx = 1;
comp_idx = 38;

wx = W_true(:,comp_idx);
ax = A_true(:,comp_idx);

[~, max_idx] = max(abs(ax));
ax = ax * sign(ax(max_idx));

clear Yenv Yseg 

% Определяем количество триалов после выборки по условию (idxs)
num_cond_trials = size(X_filt_cond, 3);

Yenv = zeros(size(X_filt_cond, 1), num_cond_trials);
Yseg = zeros(size(X_filt_cond, 1), num_cond_trials);

% 1. Корректно извлекаем компонент и его огибающую для каждой эпохи
for tr_i = 1:num_cond_trials
    % Умножаем [time x channels] на [channels x 1]
    comp_signal = X_filt_cond(:,:,tr_i) * wx; 
    
    Yseg(:,tr_i) = comp_signal;                 % Исходный сигнал источника
    Yenv(:,tr_i) = abs(hilbert(comp_signal));   % Его огибающая
end

Filt = wx;

% 3. Ограничиваем структуру электродов до 38
elec_38 = Xinf.elec;
elec_38.label = Xinf.elec.label(1:38);
elec_38.chanpos = Xinf.elec.chanpos(1:38, :);
elec_38.elecpos = Xinf.elec.elecpos(1:38, :);

figure; hold on; grid on
set(gcf,'Color','w');

% 4. Задаем оси времени
tsec_env = linspace(-1, 6, size(Yenv,1));
tsec_seg = linspace(-1, 6, size(Yseg,1));

E        = size(Yenv, 2);
env_mean = mean(Yenv, 2, 'omitnan');
sd       = std (Yenv, 0, 2, 'omitnan');

% Подготовка FieldTrip layout для 38 каналов
lay = ft_prepare_layout(struct('elec', elec_38));

% ==== ГЛАВНАЯ РАСКЛАДКА ====================================================
t = tiledlayout(3,2, 'TileSpacing','compact', 'Padding','compact');

% ---- (Left) Heatmap: все эпохи, огибающая ----
axH = nexttile(t, 1, [3 1]);          
imagesc(axH, tsec_env, 1:E, Yenv');
set(axH,'YDir','normal','Color','w'); grid(axH,'on');
xline(axH, 0, 'k--', 'LineWidth', 1);
xlabel(axH,'time, s'); ylabel(axH,'epoch');
title(axH,'Envelope per epoch');
colorbar(axH);
% caxis(axH, [0 4]); % Закомментировал, чтобы цвета автоподстроились под данные

% ---- (Right-Top) ERP без усреднения (Сигнал + Огибающая) ----
axERP = nexttile(t, 2); hold(axERP,'on'); grid(axERP,'on');
set(axERP,'Color','w');
% Рисуем сырой сигнал источника серым
plot(axERP, tsec_seg, Yseg, 'Color', [0.8 0.8 0.8]); 
% Поверх рисуем огибающие полупрозрачным синим для наглядности
plot(axERP, tsec_env, Yenv, 'Color', [0.1 0.3 0.9 0.3]); 
xline(axERP, 0, 'k--', 'LineWidth', 1);
xlabel(axERP,'time, s'); ylabel(axERP,'amplitude');
title(axERP,'Source activity & Envelopes (all epochs)');

% ---- (Right-Middle) Средняя огибающая ± SD ----
axENV = nexttile(t, 4); hold(axENV,'on'); grid(axENV,'on');
set(axENV,'Color','w');
xfill = [tsec_env, fliplr(tsec_env)];
yfill = [ (env_mean+sd).', fliplr((env_mean-sd).') ];
fill(axENV, xfill, yfill, [0.3 0.5 1.0], 'FaceAlpha',0.2, 'EdgeColor','none');
plot(axENV, tsec_env, env_mean, 'Color', [0.1 0.3 0.9], 'LineWidth', 2);
xline(axENV, 0, 'k--', 'LineWidth', 1);
xlabel(axENV,'time, s'); ylabel(axENV,'envelope (a.u.)');
title(axENV,'Mean envelope \pm SD');

% ==== (Right-Bottom) ДВА ГРАФИКА: ФИЛЬТР и ПАТТЕРН =========================
axTmp = nexttile(t, 6);
pos6  = axTmp.OuterPosition;   
delete(axTmp);

figCol = get(gcf,'Color');     
p6 = uipanel('Parent', gcf, ...
             'Units','normalized', ...
             'Position', pos6, ...
             'BorderType','none', ...
             'BackgroundColor', figCol);   
t6 = tiledlayout(p6, 1, 2, 'TileSpacing','compact', 'Padding','compact');

% --------- Левый подтайл: ТОПОГРАФИЯ ФИЛЬТРА ---------
axFiltTopo = nexttile(t6, 1);
set(axFiltTopo, 'Color','w');
valsFilt = Filt(:);
topoF = [];
topoF.dimord = 'chan_time';
topoF.label  = elec_38.label; 
topoF.time   = 0;
topoF.avg    = valsFilt;
topoF.elec   = elec_38;       
cfg = [];
cfg.figure       = axFiltTopo;
cfg.layout       = lay;
cfg.comment      = 'no';
cfg.style        = 'fill';
cfg.markersymbol = 'o';
cfg.zlim         = 'maxmin';
cfg.layout.pos(:, 1:2) = cfg.layout.pos(:, 1:2) * 1.1; 
cfg.layout.pos(:, 2) = cfg.layout.pos(:, 2) - 0.05;
ft_topoplotER(cfg, topoF);
title(axFiltTopo,'Filter');

% --------- Правый подтайл: ТОПОГРАФИЯ ПАТТЕРНА ---------
axPatTopo = nexttile(t6, 2);
set(axPatTopo, 'Color','w');
valsPat = ax(:); 
topoP = [];
topoP.dimord = 'chan_time';
topoP.label  = elec_38.label; 
topoP.time   = 0;
topoP.avg    = valsPat;
topoP.elec   = elec_38;       
cfg = [];
cfg.figure       = axPatTopo;
cfg.layout       = lay;
cfg.comment      = 'no';
cfg.style        = 'fill';
cfg.markersymbol = 'o';
cfg.zlim         = 'maxmin';
cfg.layout.pos(:, 1:2) = cfg.layout.pos(:, 1:2) * 1.1; 
cfg.layout.pos(:, 2) = cfg.layout.pos(:, 2) - 0.05;
ft_topoplotER(cfg, topoP);
title(axPatTopo,'Pattern');

% Перекрашиваем все оси внутри панели в белый
set(findall(p6, 'type','axes'), 'Color', figCol);

% ==== Синхронизация осей по X у временных графиков =========================
linkaxes([axH axERP axENV], 'x');
