close all
clear
clc

ft_path = 'C:\Users\ansbel\Documents\GitHub\CBI\site-packages\fieldtrip';
if ~exist('ft_defaults','file')
    addpath(ft_path);
end
ft_defaults;

%% Загрузка данных
elec = load("D:\OS(CURRENT)\data\simulation_support_data\eeg\elec.mat").elec;
laycfg = [];
laycfg.elec = elec;
lay = ft_prepare_layout(laycfg);     
G = load('D:\OS(CURRENT)\data\simulation_support_data\eeg\MNE_EEG_FWD_TRPL.mat').MNE_EEG_FWD_TRPL;

%% =================== ПАРАМЕТРЫ СИМУЛЯЦИИ ===================
Nsrc = 50;     
Ndistr = 1;
flanker = 1;
Ts = 100; Ntr = 50;      
Fs = 250;
Ws = 1;
Ss = 1;
nMC = 50; 
fixed_eeg_snr = 10^0.4; 
corr_range = linspace(0.1, 0.99, 12); 
nCorrLevels = length(corr_range);

labels = {'eSPoC', 'SPoC'};
nMethods = length(labels);

filcorr_train_clean = zeros(nMC, nCorrLevels, nMethods); 
filcorr_train_dirty = zeros(nMC, nCorrLevels, nMethods); 
filcorr_test_clean  = zeros(nMC, nCorrLevels, nMethods); 
filcorr_test_dirty  = zeros(nMC, nCorrLevels, nMethods); 
patcorr             = zeros(nMC, nCorrLevels, nMethods); 
target_corr         = zeros(nMC, nCorrLevels); 

for mc_idx = 1:nMC
    fprintf('Monte-Carlo iteration: %d / %d\n', mc_idx, nMC);
    
    [X_s, X_bg, X_n, z, GA, S] = generate_distributed_sources(G, Nsrc, Ndistr, flanker, Ts, Fs);
    Ainit = GA(:,1); 
    
    X = fixed_eeg_snr * X_s + X_bg + 0.1 * X_n / norm(X_s,'fro');
    X_epo = epoch_data(X', Fs, Ws, Ss);
    
    X_epo_train = X_epo(:,:,1:Ntr);
    X_epo_test  = X_epo(:,:,Ntr+1:end);
    
    nTrain = size(X_epo_train, 3);
    nTest  = size(X_epo_test, 3);
    nChan  = size(X_epo_test, 2);
    
    % Ковариации для тренировочной выборки 
    Covs_train = zeros(nChan, nChan, nTrain);
    for ep_idx = 1:nTrain
        Covs_train(:,:,ep_idx) = cov(X_epo_train(:,:,ep_idx));
    end
    
    % Ковариации для тестовой выборки 
    Covs_test = zeros(nChan, nChan, nTest);
    for ep_idx = 1:nTest
        Covs_test(:,:,ep_idx) = cov(X_epo_test(:,:,ep_idx));
    end
    
    z_epo_raw = epoch_data(z(1,:)', Fs, Ws, Ss); 
    z_epo = squeeze(mean(z_epo_raw, 1));         
    z_epo = (z_epo - mean(z_epo)) / std(z_epo);
    
    % Строго стандартизируем трейн и тест независимо 
    z_train = z_epo(1:Ntr);
    z_train = (z_train - mean(z_train)) / std(z_train);
    
    z_test  = z_epo(Ntr+1:end);
    z_test  = (z_test - mean(z_test)) / std(z_test);
    
    % Локальные переменные для parfor
    f_tr_clean_loc = zeros(nCorrLevels, nMethods); 
    f_tr_dirty_loc = zeros(nCorrLevels, nMethods); 
    f_te_clean_loc = zeros(nCorrLevels, nMethods); 
    f_te_dirty_loc = zeros(nCorrLevels, nMethods); 
    p_corr_loc     = zeros(nCorrLevels, nMethods);
    t_corr_loc     = zeros(nCorrLevels, 1);
    
    parfor corr_idx = 1:nCorrLevels
        current_r = corr_range(corr_idx);
        
        % === ГЕНЕРАЦИЯ ШУМА ДЛЯ ТРЕЙНА ===
        n_raw_train = randn(1, nTrain);
        n_orth_train = n_raw_train - (n_raw_train * z_train') / (z_train * z_train') * z_train;
        n_orth_train = (n_orth_train - mean(n_orth_train)) / std(n_orth_train);
        z_noisy_train = current_r * z_train + sqrt(1 - current_r^2) * n_orth_train;
        z_noisy_train = (z_noisy_train - mean(z_noisy_train)) / std(z_noisy_train); 
        
        % === ГЕНЕРАЦИЯ ШУМА ДЛЯ ТЕСТА ===
        n_raw_test = randn(1, nTest);
        n_orth_test = n_raw_test - (n_raw_test * z_test') / (z_test * z_test') * z_test;
        n_orth_test = (n_orth_test - mean(n_orth_test)) / std(n_orth_test);
        z_noisy_test = current_r * z_test + sqrt(1 - current_r^2) * n_orth_test;
        z_noisy_test = (z_noisy_test - mean(z_noisy_test)) / std(z_noisy_test); 
        
        t_corr_loc(corr_idx) = abs(corr(z_noisy_train', z_train'));
        
        w_all = zeros(nChan, nMethods);
        a_all = zeros(nChan, nMethods);
        
        % ================= 1. eSPoC =================
        [W_e, A_e] = espoc(X_epo_train, z_noisy_train);
        if size(W_e, 3) > 1 || size(W_e, 1) == 1, W_e = squeeze(W_e(1,:,:)); A_e = squeeze(A_e(1,:,:)); end
        if ~isempty(W_e), w_all(:, 1) = W_e(:,1); a_all(:, 1) = A_e(:,1); end
        
        % ================= 2. SPoC =================
        [W_s, A_s] = spoc(X_epo_train, z_noisy_train);
        if ~isempty(W_s), w_all(:, 2) = W_s(:,1); a_all(:, 2) = A_s(:,1); end
        
        % Проверка метрик
        for m_idx = 1:nMethods
            w = w_all(:, m_idx); a = a_all(:, m_idx);
            
            env_train = zeros(nTrain, 1);
            for ep_idx = 1:nTrain, env_train(ep_idx) = w' * Covs_train(:,:,ep_idx) * w; end
            
            env_test = zeros(nTest, 1);
            for ep_idx = 1:nTest, env_test(ep_idx) = w' * Covs_test(:,:,ep_idx) * w; end
            
            % Сбор 4-х метрик
            f_tr_clean_loc(corr_idx, m_idx) = abs(corr(env_train(:), z_train(:)));
            f_tr_dirty_loc(corr_idx, m_idx) = abs(corr(env_train(:), z_noisy_train(:)));
            
            f_te_clean_loc(corr_idx, m_idx) = abs(corr(env_test(:), z_test(:)));
            f_te_dirty_loc(corr_idx, m_idx) = abs(corr(env_test(:), z_noisy_test(:)));
            
            p_corr_loc(corr_idx, m_idx)     = abs(corr(a, Ainit));
        end
    end
    
    filcorr_train_clean(mc_idx, :, :) = f_tr_clean_loc;
    filcorr_train_dirty(mc_idx, :, :) = f_tr_dirty_loc;
    filcorr_test_clean(mc_idx, :, :)  = f_te_clean_loc;
    filcorr_test_dirty(mc_idx, :, :)  = f_te_dirty_loc;
    patcorr(mc_idx, :, :)             = p_corr_loc;
    target_corr(mc_idx, :)            = t_corr_loc';
end

%% Вычисление статистики и отрисовка
x_sorted = mean(target_corr, 1); 

% Расчет средних и ДИ
get_mean = @(arr) squeeze(mean(arr, 1));
get_ci   = @(arr) squeeze(1.96 * std(arr, 0, 1) / sqrt(nMC));

mean_tr_cl = get_mean(filcorr_train_clean); ci_tr_cl = get_ci(filcorr_train_clean);
mean_tr_dt = get_mean(filcorr_train_dirty); ci_tr_dt = get_ci(filcorr_train_dirty);
mean_te_cl = get_mean(filcorr_test_clean);  ci_te_cl = get_ci(filcorr_test_clean);
mean_te_dt = get_mean(filcorr_test_dirty);  ci_te_dt = get_ci(filcorr_test_dirty);
mean_p     = get_mean(patcorr);             ci_p     = get_ci(patcorr);

% Отрисовка
figure('Position', [50 50 1600 800], 'Color', 'w'); 
colors = [0.8 0 0; 0 0.7 0]; 

% Академичные заголовки для статьи
titles = {
    'Train Data: Extracted vs Noisy Target', ...
    'Train Data: Extracted vs True Source', ...
    'Test Data: Extracted vs Noisy Target', ...
    'Test Data: Extracted vs True Source'
};

y_data = {mean_tr_dt, mean_tr_cl, mean_te_dt, mean_te_cl};
ci_data = {ci_tr_dt, ci_tr_cl, ci_te_dt, ci_te_cl};

% Отрисовка 4 графиков ковариаций
for p = 1:4
    subplot(2, 3, p + (p>2)); % Размещаем их сеткой 2x2, оставляя правую колонку пустой
    hold on;
    for m = 1:nMethods
        y = y_data{p}(:, m)'; ci = ci_data{p}(:, m)';
        fill([x_sorted fliplr(x_sorted)], [y-ci fliplr(y+ci)], colors(m,:), 'FaceAlpha', 0.15, 'EdgeColor', 'none', 'HandleVisibility', 'off');   
        plot(x_sorted, y, '-o', 'Color', colors(m,:), 'LineWidth', 2, 'MarkerSize', 5, 'MarkerFaceColor', 'w', 'DisplayName', labels{m});
    end
    title(titles{p}, 'FontSize', 12, 'FontWeight', 'bold');
    xlabel('Target-to-Source Correlation (r)', 'FontSize', 11); 
    ylabel('Correlation coefficient (r)', 'FontSize', 11);
    ylim([0 1.05]); xlim([min(x_sorted) max(x_sorted)]); grid on; 
    if p==1, legend('Location', 'northwest'); end
end

% Отрисовка паттернов (справа по центру)
subplot(2, 3, [3 6]); hold on; % Занимает всю правую колонку
for m = 1:nMethods
    y = mean_p(:, m)'; ci = ci_p(:, m)';
    fill([x_sorted fliplr(x_sorted)], [y-ci fliplr(y+ci)], colors(m,:), 'FaceAlpha', 0.15, 'EdgeColor', 'none', 'HandleVisibility', 'off');   
    plot(x_sorted, y, '-o', 'Color', colors(m,:), 'LineWidth', 2, 'MarkerSize', 5, 'MarkerFaceColor', 'w', 'DisplayName', labels{m});
end
title('Spatial Pattern Recovery', 'FontSize', 12, 'FontWeight', 'bold');
xlabel('Target-to-Source Correlation (r)', 'FontSize', 11); 
ylabel('Pattern Correlation (r)', 'FontSize', 11);
ylim([0 1.05]); xlim([min(x_sorted) max(x_sorted)]); grid on; legend('Location', 'northwest');