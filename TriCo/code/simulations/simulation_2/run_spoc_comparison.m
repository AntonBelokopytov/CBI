close all
clear
clc

ft_path = 'C:\Users\anton\Documents\GitHub\CBI\site-packages\fieldtrip';
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
Ts = 850;      
Fs = 250;
Ws = 1;
Ss = 1;
nMC = 100; 
  
SNR_range = 10.^(-1.4 : 0.2 : 1); 
nSNR = length(SNR_range);

labels = {'eSPoC', 'mSPoC', 'PCA + SPoC'};
nMethods = length(labels);

% Массивы для метрик
filcorr_test = zeros(nMC, nSNR, nMethods); 
patcorr      = zeros(nMC, nSNR, nMethods); 

for mc_idx = 1:nMC
    fprintf('Monte-Carlo iteration: %d / %d\n', mc_idx, nMC);
    
    % 1. Генерируем "чистые" источники
    [X_s, X_bg, X_n, z, GA, S] = generate_distributed_sources(G, Nsrc, Ndistr, flanker, Ts, Fs);
    Ainit = GA(:,1); 
    
    % Извлекаем огибающую
    z_epo_raw = epoch_data(z(1,:)', Fs, Ws, Ss); 
    z_epo = squeeze(mean(z_epo_raw, 1));         
    z_epo = (z_epo - mean(z_epo)) / std(z_epo);
    
    z_train = z_epo(1:250);
    z_test  = z_epo(251:end);
    
    n_ext = 5;
    ext_weights = rand(n_ext, 1) * 2 - 1; 
    noise_level = 1.0; 
    
    f_test_local = zeros(nSNR, nMethods); 
    p_corr_local = zeros(nSNR, nMethods);
    
    % 2. Перебираем разные уровни SNR
    parfor snr_idx = 1:nSNR
        current_SNR = SNR_range(snr_idx);
        
        X = current_SNR * X_s + X_bg + 0.1 * X_n / norm(X_s,'fro');
        X_epo = epoch_data(X', Fs, Ws, Ss);
        
        X_epo_train = X_epo(:,:,1:250);
        X_epo_test  = X_epo(:,:,251:end);
        
        nTrain = size(X_epo_train, 3);
        nTest  = size(X_epo_test, 3);
        nChan  = size(X_epo_test, 2);
        
        Covs_train = zeros(nChan, nChan, nTrain);
        for ep_idx = 1:nTrain
            Covs_train(:,:,ep_idx) = cov(X_epo_train(:,:,ep_idx));
        end
        
        Covs_test = zeros(nChan, nChan, nTest);
        for ep_idx = 1:nTest
            Covs_test(:,:,ep_idx) = cov(X_epo_test(:,:,ep_idx));
        end
        
        % Внешние переменные
        z_multidim_train = ext_weights * z_train + noise_level * randn(n_ext, nTrain);
        
        [~, score] = pca(z_multidim_train');
        z_spoc_train = score(:,1)';
        
        w_all = zeros(nChan, nMethods);
        a_all = zeros(nChan, nMethods);
        
        % ================= 1. eSPoC =================
        [W_e, A_e] = espoc(X_epo_train, z_multidim_train);
        if size(W_e, 3) > 1 || size(W_e, 1) == 1, W_e = squeeze(W_e(1,:,:)); A_e = squeeze(A_e(1,:,:)); end
        if ~isempty(W_e), w_all(:, 1) = W_e(:,1); a_all(:, 1) = A_e(:,1); end
        
        % ================= 2. mSPoC =================
        mspoc_opts = struct('tau_vector', 0, 'n_component_sets', 1, 'verbose', 0);
        [W_m, ~, ~, A_m, ~] = mspoc(X_epo_train, z_multidim_train, mspoc_opts);
        if ~isempty(W_m), w_all(:, 2) = W_m(:,1); a_all(:, 2) = A_m(:,1); end
        
        % ================= 3. SPoC =================
        [W_s, A_s] = spoc(X_epo_train, z_spoc_train);
        if ~isempty(W_s), w_all(:, 3) = W_s(:,1); a_all(:, 3) = A_s(:,1); end
        
        % Проверка на тесте
        for m_idx = 1:nMethods
            w = w_all(:, m_idx); a = a_all(:, m_idx);
            
            env_test = zeros(nTest, 1);
            for ep_idx = 1:nTest
                env_test(ep_idx) = w' * Covs_test(:,:,ep_idx) * w;
            end
            
            f_test_local(snr_idx, m_idx) = abs(corr(env_test(:), z_test(:)));
            p_corr_local(snr_idx, m_idx) = abs(corr(a, Ainit));
        end
    end
    
    filcorr_test(mc_idx, :, :) = f_test_local;
    patcorr(mc_idx, :, :)      = p_corr_local;
end

%% Вычисление статистики и отрисовка
mean_f_test = squeeze(mean(filcorr_test, 1));
mean_p_corr = squeeze(mean(patcorr, 1));   
ci_f_test = squeeze(1.96 * std(filcorr_test, 0, 1) / sqrt(nMC));
ci_p_corr = squeeze(1.96 * std(patcorr, 0, 1) / sqrt(nMC));

x = SNR_range; 
figure('Position', [100 100 950 450], 'Color', 'w'); 
colors = [0.8 0 0; 0 0 0.8; 0 0.7 0]; 

subplot(1,2,1); hold on;
for m = 1:nMethods
    y  = mean_f_test(:,m)'; ci = ci_f_test(:,m)';
    fill([x fliplr(x)], [y-ci fliplr(y+ci)], colors(m,:), 'FaceAlpha', 0.15, 'EdgeColor', 'none', 'HandleVisibility', 'off');   
    semilogx(x, y, '-o', 'Color', colors(m,:), 'LineWidth', 2, 'MarkerSize', 5, 'MarkerFaceColor', 'w', 'DisplayName', labels{m});
end
title('TEST: Correlation with Source Power', 'FontSize', 12, 'FontWeight', 'bold');
xlabel('Signal-to-Noise Ratio (SNR)', 'FontSize', 11); ylabel('Correlation, r', 'FontSize', 11);
ylim([0 1.05]); xlim([min(x) max(x)]); grid on; set(gca, 'XScale', 'log'); legend('Location', 'northwest');

subplot(1,2,2); hold on;
for m = 1:nMethods
    y  = mean_p_corr(:,m)'; ci = ci_p_corr(:,m)';
    fill([x fliplr(x)], [y-ci fliplr(y+ci)], colors(m,:), 'FaceAlpha', 0.15, 'EdgeColor', 'none', 'HandleVisibility', 'off');   
    semilogx(x, y, '-o', 'Color', colors(m,:), 'LineWidth', 2, 'MarkerSize', 5, 'MarkerFaceColor', 'w', 'DisplayName', labels{m});
end
title('Spatial Pattern Correlation', 'FontSize', 12, 'FontWeight', 'bold');
xlabel('Signal-to-Noise Ratio (SNR)', 'FontSize', 11); ylabel('Correlation, r', 'FontSize', 11);
ylim([0 1.05]); xlim([min(x) max(x)]); grid on; set(gca, 'XScale', 'log'); legend('Location', 'northwest');