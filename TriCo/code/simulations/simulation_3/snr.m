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
Nsrc = 100;     
Ndistr = 1;
flanker = 1;
Ts = 850; Ntr = 250;      
Fs = 250;
Ws = 1;
Ss = 1;
nMC = 50; 

% Задаем сетку SNR и фиксируем корреляцию таргета
snr_range = logspace(-1.4, 1, 12);
nSNRLevels = length(snr_range);
fixed_target_corr = 0.8;

labels = {'eSPoCtcsp', 'SPoC'};
nMethods = length(labels);

% Массивы для хранения результатов
filcorr_train_clean = zeros(nMC, nSNRLevels, nMethods); 
filcorr_train_dirty = zeros(nMC, nSNRLevels, nMethods); 
filcorr_test_clean  = zeros(nMC, nSNRLevels, nMethods); 
patcorr             = zeros(nMC, nSNRLevels, nMethods); 

for mc_idx = 1:nMC
    fprintf('Monte-Carlo iteration: %d / %d\n', mc_idx, nMC);
    
    [X_s, X_bg, X_n, z, GA, ~] = generate_distributed_sources(G, Nsrc, Ndistr, flanker, Ts, Fs);
    Ainit = GA(:,1); 
    
    % Подготовка идеального таргета
    z_epo_raw = epoch_data(z(1,:)', Fs, Ws, Ss); 
    z_epo = squeeze(mean(z_epo_raw, 1));         
    z_epo = (z_epo - mean(z_epo)) / std(z_epo);
    
    z_train = z_epo(1:Ntr);
    z_train = (z_train - mean(z_train)) / std(z_train);
    
    z_test  = z_epo(Ntr+1:end);
    z_test  = (z_test - mean(z_test)) / std(z_test);
    
    nTrain = length(z_train);
    nTest  = length(z_test);
    nChan  = size(G, 1);
    
    % === ГЕНЕРАЦИЯ ФИКСИРОВАННОГО ШУМА ДЛЯ ТАРГЕТА ===
    n_raw_train = randn(1, nTrain);
    n_orth_train = n_raw_train - (n_raw_train * z_train') / (z_train * z_train') * z_train;
    n_orth_train = (n_orth_train - mean(n_orth_train)) / std(n_orth_train);
    z_noisy_train = fixed_target_corr * z_train + sqrt(1 - fixed_target_corr^2) * n_orth_train;
    z_noisy_train = (z_noisy_train - mean(z_noisy_train)) / std(z_noisy_train); 
    
    % Локальные переменные для parfor
    f_tr_clean_loc = zeros(nSNRLevels, nMethods); 
    f_tr_dirty_loc = zeros(nSNRLevels, nMethods); 
    f_te_clean_loc = zeros(nSNRLevels, nMethods); 
    p_corr_loc     = zeros(nSNRLevels, nMethods);
    
    parfor snr_idx = 1:nSNRLevels
        current_snr = snr_range(snr_idx);

        % Смешивание с текущим SNR
        X = current_snr * X_s + X_bg + 0.1 * X_n / norm(X_s,'fro');
        
        % Нарезка на эпохи внутри цикла, так как сигнал изменился
        X_epo = epoch_data(X', Fs, Ws, Ss);
        X_epo_train = X_epo(:,:,1:nTrain);
        X_epo_test  = X_epo(:,:,nTrain+1:end);
        
        % Ковариации
        Covs_train = zeros(nChan, nChan, nTrain);
        for ep_idx = 1:nTrain, Covs_train(:,:,ep_idx) = cov(X_epo_train(:,:,ep_idx)); end
        
        Covs_test = zeros(nChan, nChan, nTest);
        for ep_idx = 1:nTest, Covs_test(:,:,ep_idx) = cov(X_epo_test(:,:,ep_idx)); end
        
        w_all = zeros(nChan, nMethods);
        a_all = zeros(nChan, nMethods);
        
        % ================= 1. eSPoC =================
        [W_e, A_e] = espoct_csp(X_epo_train, z_noisy_train);
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
            
            % Сбор метрик
            f_tr_clean_loc(snr_idx, m_idx) = abs(corr(env_train(:), z_train(:)));
            f_tr_dirty_loc(snr_idx, m_idx) = abs(corr(env_train(:), z_noisy_train(:)));
            f_te_clean_loc(snr_idx, m_idx) = abs(corr(env_test(:), z_test(:)));
            p_corr_loc(snr_idx, m_idx)     = abs(corr(a, Ainit));
        end
    end
    
    filcorr_train_clean(mc_idx, :, :) = f_tr_clean_loc;
    filcorr_train_dirty(mc_idx, :, :) = f_tr_dirty_loc;
    filcorr_test_clean(mc_idx, :, :)  = f_te_clean_loc;
    patcorr(mc_idx, :, :)             = p_corr_loc;
end

%% =================== ВЫЧИСЛЕНИЕ СТАТИСТИКИ И ОТРИСОВКА ===================
x_sorted = snr_range; 

get_mean = @(arr) squeeze(mean(arr, 1));
get_ci   = @(arr) squeeze(1.96 * std(arr, 0, 1) / sqrt(nMC));

mean_tr_dt = get_mean(filcorr_train_dirty); ci_tr_dt = get_ci(filcorr_train_dirty);
mean_tr_cl = get_mean(filcorr_train_clean); ci_tr_cl = get_ci(filcorr_train_clean);
mean_te_cl = get_mean(filcorr_test_clean);  ci_te_cl = get_ci(filcorr_test_clean);
mean_p     = get_mean(patcorr);             ci_p     = get_ci(patcorr);

figure('Position', [100 100 1200 800], 'Color', 'w');
colors = [0.8 0 0; 0 0.7 0]; 

titles = {
    sprintf('Train Data: Extracted vs Noisy Target (Target r=%.1f)', fixed_target_corr), ...
    'Train Data: Extracted vs True Source', ...
    'Test Data: Extracted vs True Source'
};

y_data = {mean_tr_dt, mean_tr_cl, mean_te_cl};
ci_data = {ci_tr_dt, ci_tr_cl, ci_te_cl};

for p = 1:3
    subplot(2, 2, p);
    hold on;
    for m = 1:nMethods
        y = y_data{p}(:, m)'; ci = ci_data{p}(:, m)';
        fill([x_sorted fliplr(x_sorted)], [y-ci fliplr(y+ci)], colors(m,:), 'FaceAlpha', 0.15, 'EdgeColor', 'none', 'HandleVisibility', 'off');   
        plot(x_sorted, y, '-o', 'Color', colors(m,:), 'LineWidth', 2, 'MarkerSize', 5, 'MarkerFaceColor', 'w', 'DisplayName', labels{m});
    end
    title(titles{p}, 'FontSize', 12, 'FontWeight', 'bold');
    xlabel('Signal-to-Noise Ratio (SNR)', 'FontSize', 11); 
    ylabel('Correlation coefficient (r)', 'FontSize', 11);
    ylim([0 1.05]); xlim([min(x_sorted) max(x_sorted)]); grid on; 
    set(gca, 'XScale', 'log'); % Логарифмическая шкала для SNR
    if p==1, legend('Location', 'northwest'); end
end

subplot(2, 2, 4); 
hold on; 
for m = 1:nMethods
    y = mean_p(:, m)'; ci = ci_p(:, m)';
    fill([x_sorted fliplr(x_sorted)], [y-ci fliplr(y+ci)], colors(m,:), 'FaceAlpha', 0.15, 'EdgeColor', 'none', 'HandleVisibility', 'off');   
    plot(x_sorted, y, '-o', 'Color', colors(m,:), 'LineWidth', 2, 'MarkerSize', 5, 'MarkerFaceColor', 'w', 'DisplayName', labels{m});
end
title('Spatial Pattern Recovery', 'FontSize', 12, 'FontWeight', 'bold');
xlabel('Signal-to-Noise Ratio (SNR)', 'FontSize', 11); 
ylabel('Pattern Correlation (r)', 'FontSize', 11);
ylim([0 1.05]); xlim([min(x_sorted) max(x_sorted)]); grid on;
set(gca, 'XScale', 'log'); % Логарифмическая шкала для SNR