close all
clear
clc

ft_path = 'C:\Users\ansbel\Documents\2Git\fieldtrip';

if ~exist('ft_defaults','file')
    addpath(ft_path);
end

ft_defaults;

%%
elec = load("elec.mat").elec;

topo = [];
topo.dimord = 'chan_time';
topo.label  = elec.label;  
topo.time   = 0;
topo.elec   = elec;

laycfg = [];
laycfg.elec = elec;
lay = ft_prepare_layout(laycfg);     

cfg = [];
cfg.marker       = '';
cfg.layout       = lay;
cfg.comment      = 'no';
cfg.style        = 'fill';
cfg.markersymbol = '';
cfg.colorbar     = 'no'; 
cfg.layout.pos(:, 1:2) = cfg.layout.pos(:, 1:2) * 1.1; 
cfg.layout.pos(:, 2) = cfg.layout.pos(:, 2) - 0.05;

%% Настройки симуляции
G = load('MNE_EEG_FWD_TRPL.mat').MNE_EEG_FWD_TRPL;
Nsrc = 101;
Ndistr = 1;
flanker = 1;
Fs = 250;

% === ПАРАМЕТРЫ ===
nMC = 10;                           % Число итераций Монте-Карло
SNR = 10^(0.4);                    % ФИКСИРОВАННЫЙ уровень шума (например, -10 дБ)
nTr_range = [50, 100, 200:500:8000]; % Варианты длины обучающей выборки
nTr_len = length(nTr_range);
nTr_max = max(nTr_range);           % Максимальная длина трейна (чтобы сдвинуть тест)
nTe = 1000;                         % Фиксированная длина тестовой выборки
Ts = nTr_max + nTe;                 % Общее время генерации сигналов

% Инициализация матриц результатов (размеры: [nMC x nTr_len])
corr_gl_train = zeros(nMC, nTr_len);
corr_gl_test  = zeros(nMC, nTr_len);
corr_lcl_train= zeros(nMC, nTr_len);
corr_lcl_test = zeros(nMC, nTr_len);
patcorr       = zeros(nMC, nTr_len);

% Основной цикл
for mc_idx = 1:nMC
    disp(['Monte-Carlo iteration: ', num2str(mc_idx)])
    
    % =================
    % 1. Data generation (ОДИН РАЗ НА ИТЕРАЦИЮ)
    % =================
    [X_s, X_bg, X_n, z, GA, S] = generate_distributed_sources( ...
        G, Nsrc, Ndistr, flanker, Ts, Fs);
    
    X = SNR*X_s + X_bg + 0.1 * X_n / norm(X_s,'fro');
    
    % Эпохирование всего массива
    X_epo = epoch_data(X', Fs, 1, 1);
    z_epo = squeeze(mean(epoch_data(z(1,:)', Fs, 1, 1), 1));
    
    nTotal = size(X_epo,3);
    nChan = size(X_epo,2);
    nFeat = nChan*(nChan+1)/2;
    
    % =================
    % 2. Precompute covariance matrices 
    % =================
    Covs_all = zeros(nChan, nChan, nTotal);
    Feat_all = zeros(nFeat, nTotal);
    for ep_idx = 1:nTotal
        C = cov(X_epo(:,:,ep_idx));
        Covs_all(:,:,ep_idx) = C;
        Feat_all(:,ep_idx) = cov2upper(C);
    end
    
    % =================
    % 3. Фиксированная тестовая выборка (ВСЕГДА ОДНА И ТА ЖЕ ДЛЯ ЧЕСТНОСТИ)
    % =================
    idx_test = nTr_max + 1 : nTr_max + nTe;
    X_epo_test = X_epo(:,:,idx_test);
    z_epo_test = z_epo(idx_test);
    Covs_test  = Covs_all(:,:,idx_test);
    Feat_test  = Feat_all(:,idx_test);
    
    % =================
    % 4. Цикл по размеру обучающей выборки
    % =================
    parfor tr_idx = 1:nTr_len
        nTr = nTr_range(tr_idx);
        
        % Берем первые nTr эпох для обучения
        idx_train = 1:nTr;
        X_epo_train = X_epo(:,:,idx_train);
        z_epo_train = z_epo(idx_train);
        Covs_train  = Covs_all(:,:,idx_train);
        Feat_train  = Feat_all(:,idx_train);
        
        % ===== eSPoC =====
        % Не забываем транспонировать z_epo_train для espoc
        [W, A, Vf, ~, corrs] = espoc(X_epo_train, z_epo_train);
        w = W(:,1);
        
        % Вычисление огибающих на тестовой выборке
        env_test = zeros(nTe,1);
        for ep_idx = 1:nTe
            env_test(ep_idx) = w' * Covs_test(:,:,ep_idx) * w;
        end
        
        gl_env_train = Vf(:,1)' * Feat_train;
        gl_env_test  = Vf(:,1)' * Feat_test;
        
        % Вычисление метрик (с модулями abs, так как знак CCA произволен)
        corr_gl_train_loc = abs(corr(gl_env_train', z_epo_train'));
        corr_gl_test_loc  = abs(corr(gl_env_test', z_epo_test'));
        corr_lcl_train_loc= abs(corrs(1));
        corr_lcl_test_loc = abs(corr(env_test, z_epo_test'));
        patcorr_loc       = abs(corr(GA(:,1), A(:,1)));
        
        % Запись в общие массивы
        corr_gl_train(mc_idx, tr_idx) = corr_gl_train_loc;
        corr_gl_test(mc_idx,  tr_idx) = corr_gl_test_loc;
        corr_lcl_train(mc_idx,tr_idx) = corr_lcl_train_loc;
        corr_lcl_test(mc_idx, tr_idx) = corr_lcl_test_loc;
        patcorr(mc_idx,       tr_idx) = patcorr_loc;
    end
end

%% Вычисление статистики и построение графиков
% =================
% Statistics
% =================
% Means
mean_gl_train = mean(corr_gl_train, 1);
mean_gl_test  = mean(corr_gl_test, 1);
mean_lcl_train= mean(corr_lcl_train, 1);
mean_lcl_test = mean(corr_lcl_test, 1);
mean_pat      = mean(patcorr, 1);

% 95% CI
ci_gl_train = 1.96 * std(corr_gl_train, 0, 1) / sqrt(nMC);
ci_gl_test  = 1.96 * std(corr_gl_test, 0, 1)  / sqrt(nMC);
ci_lcl_train= 1.96 * std(corr_lcl_train, 0, 1) / sqrt(nMC);
ci_lcl_test = 1.96 * std(corr_lcl_test, 0, 1)  / sqrt(nMC);
ci_pat      = 1.96 * std(patcorr, 0, 1) / sqrt(nMC);

% =================
% Plot Setup
% =================
x = nTr_range;
xticks_vals = nTr_range;
% Автоматически создаем подписи оси X из значений nTr_range
xticks_lbls = arrayfun(@num2str, nTr_range, 'UniformOutput', false);

figure('Position', [100 100 1000 400], 'Color', 'w');

% =================
% Envelope correlation
% =================
subplot(1,2,1)
hold on
colors = [
    0 0.3 1  % Global train (blue)
    0 0.3 1  % Global test (blue)
    1 0 0    % Local train (red)
    1 0 0    % Local test (red)
];
styles = {'--', '-', '--', '-'};
markers = {'none', 'o', 'none', 'o'}; % Маркеры только для тестовых кривых

means = {mean_gl_train, mean_gl_test, mean_lcl_train, mean_lcl_test};
cis   = {ci_gl_train, ci_gl_test, ci_lcl_train, ci_lcl_test};
labels = {'Global train', 'Global test', 'Local train', 'Local test'};

h_lines = gobjects(4,1); % Массив объектов для корректной легенды

for i = 1:4
    y  = means{i};
    ci = cis{i};
    
    % Закрашенный доверительный интервал
    fill([x fliplr(x)], ...
         [y-ci fliplr(y+ci)], ...
         colors(i,:), ...
         'FaceAlpha', 0.2, ...
         'EdgeColor', 'none', ...
         'HandleVisibility', 'off');
     
    % Линия графика (заменили semilogx на plot)
    h_lines(i) = plot(x, y, ...
        'Color', colors(i,:), ...
        'LineStyle', styles{i}, ...
        'LineWidth', 2, ...
        'Marker', markers{i}, ...
        'MarkerFaceColor', 'w');
end

title('Envelope Correlation vs Sample Size')
xlabel('Number of training epochs')
ylabel('Correlation')
ylim([0 1])
grid on
ax = gca;
ax.XMinorGrid = 'on';
ax.XTick = xticks_vals;
ax.XTickLabel = xticks_lbls;
legend(h_lines, labels, 'Location', 'southeast')

% =================
% Pattern correlation
% =================
subplot(1,2,2)
hold on

% Закрашенный доверительный интервал
fill([x fliplr(x)], ...
     [mean_pat-ci_pat fliplr(mean_pat+ci_pat)], ...
     [1 0 0], ...
     'FaceAlpha', 0.25, ...
     'EdgeColor', 'none', ...
     'HandleVisibility', 'off');

% Линия графика (заменили semilogx на plot)
h_pat = plot(x, mean_pat, ...
    'Color', [1 0 0], ...
    'LineWidth', 2, ...
    'Marker', 'o', ...
    'MarkerFaceColor', 'w');

title('Pattern Correlation vs Sample Size')
xlabel('Number of training epochs')
ylabel('Correlation')
ylim([0 1])
grid on
ax = gca;
ax.XMinorGrid = 'on';
ax.XTick = xticks_vals;
ax.XTickLabel = xticks_lbls;
legend(h_pat, {'Pattern'}, 'Location', 'southeast')
