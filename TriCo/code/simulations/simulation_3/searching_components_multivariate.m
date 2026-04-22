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
Ndistr = 2; 
flanker = 1;
Ts = 850; Ntr = 250;      
Fs = 250; Ws = 1; Ss = 1;
nMC = 10; 

% ТЕПЕРЬ МЫ ВАРЬИРУЕМ SNR ЭЭГ (от очень слабого сигнала до сильного)
snr_range = logspace(-1.5, 0.5, 8); 
nSNR = length(snr_range);
fixed_target_noise = 1.0; % Умеренно шумный многомерный таргет

labels = {'eSPoC', 'mSPoC', 'PCA + SPoC'};
nMethods = length(labels);

% Массивы для метрик
patcorr_1 = zeros(nMC, nSNR, nMethods); 
patcorr_2 = zeros(nMC, nSNR, nMethods); 
filcorr_test_1 = zeros(nMC, nSNR, nMethods); 
filcorr_test_2 = zeros(nMC, nSNR, nMethods); 
time_comp = zeros(nMC, nSNR, nMethods); 

for mc_idx = 1:nMC
    fprintf('Monte-Carlo iteration: %d / %d\n', mc_idx, nMC);
    
    % Генерируем источники (не зависят от SNR)
    [X_s, X_bg, X_n, z, GA, S] = generate_distributed_sources(G, Nsrc, Ndistr, flanker, Ts, Fs);
    Ainit = GA(:, 1:2); 
    
    % Истинные огибающие
    z_epo_raw = epoch_data(z(1:2,:)', Fs, Ws, Ss); 
    z_epo = squeeze(mean(z_epo_raw, 1))';         
    z_train = z_epo(1:Ntr, :); 
    z_train = (z_train - mean(z_train, 1)) ./ std(z_train, 0, 1);
    z_test = z_epo(Ntr+1:end, :);
    z_test = (z_test - mean(z_test, 1)) ./ std(z_test, 0, 1);
    
    nTrain = size(z_train, 1);
    nTest  = size(z_test, 1);
    
    % Локальные переменные для parfor
    p_corr_1_loc = zeros(nSNR, nMethods);
    p_corr_2_loc = zeros(nSNR, nMethods);
    f_test_1_loc = zeros(nSNR, nMethods);
    f_test_2_loc = zeros(nSNR, nMethods);
    time_loc     = zeros(nSNR, nMethods);
    
    n_ext = 5;
    mix_mat = randn(n_ext, n_ext);
    
    Z_multi_train = zeros(nTrain, n_ext);
    Z_multi_train(:, 1:2) = z_train;
    for d = 3:n_ext
        nd = randn(nTrain, 1);
        nd = nd - z_train(:,1)*(z_train(:,1)'*nd)/(z_train(:,1)'*z_train(:,1)) ...
                - z_train(:,2)*(z_train(:,2)'*nd)/(z_train(:,2)'*z_train(:,2));
        Z_multi_train(:, d) = (nd - mean(nd)) / std(nd) * fixed_target_noise;
    end
    Z_noisy_train_multi = (mix_mat * Z_multi_train')';
    
    % PCA для SPoC
    [~, score_tr] = pca(Z_noisy_train_multi);
    z_spoc_train = score_tr(:, 1:2);
    
    parfor snr_idx = 1:nSNR
        current_snr = snr_range(snr_idx);
        
        X = current_snr * X_s + X_bg + 0.1 * X_n / norm(X_s,'fro');
        X_epo = epoch_data(X', Fs, Ws, Ss);
        
        X_epo_train = X_epo(:,:,1:Ntr);
        X_epo_test  = X_epo(:,:,Ntr+1:end);
        nChan  = size(X_epo_train, 2);
        
        Covs_test = zeros(nChan, nChan, nTest);
        for ep_idx = 1:nTest, Covs_test(:,:,ep_idx) = cov(X_epo_test(:,:,ep_idx)); end
        
        A_est = cell(1, nMethods);
        W_est = cell(1, nMethods);
        t_elapsed = zeros(1, nMethods);
        
        % ================= 1. eSPoC =================
        tic;
        [W_e, A_e] = espoc(X_epo_train, Z_noisy_train_multi', 'X_min_var_explained', 0.90);
        A_est{1} = squeeze(A_e(1, :, 1:2)); 
        W_est{1} = squeeze(W_e(1, :, 1:2)); 
        t_elapsed(1) = toc;
        
        % ================= 2. mSPoC =================
        tic;
        mspoc_opts = struct('n_component_sets', 2, 'verbose', 0);
        [W_m, ~, ~, A_m, ~] = mspoc(X_epo_train, Z_noisy_train_multi', mspoc_opts);
        t_elapsed(2) = toc;
        A_est{2} = A_m(:, 1:2);
        W_est{2} = W_m(:, 1:2);
        
        % ================= 3. PCA + SPoC =================
        tic;
        [W_s1, A_s1] = spoc(X_epo_train, z_spoc_train(:,1)');
        [W_s2, A_s2] = spoc(X_epo_train, z_spoc_train(:,2)');
        t_elapsed(3) = toc;
        A_est{3} = [A_s1(:,1), A_s2(:,1)];
        W_est{3} = [W_s1(:,1), W_s2(:,1)];
        
        % Оценка качества
        for m_idx = 1:nMethods
            C_A = abs(corr(A_est{m_idx}, Ainit)); 
            p_corr_1_loc(snr_idx, m_idx) = max(C_A(:, 1));
            p_corr_2_loc(snr_idx, m_idx) = max(C_A(:, 2));
            
            W_curr = W_est{m_idx};
            env_test_1 = zeros(nTest, 1);
            env_test_2 = zeros(nTest, 1);
            for ep_idx = 1:nTest
                env_test_1(ep_idx) = W_curr(:,1)' * Covs_test(:,:,ep_idx) * W_curr(:,1);
                env_test_2(ep_idx) = W_curr(:,2)' * Covs_test(:,:,ep_idx) * W_curr(:,2);
            end
            
            C_env = abs(corr([env_test_1, env_test_2], z_test));
            f_test_1_loc(snr_idx, m_idx) = max(C_env(:, 1));
            f_test_2_loc(snr_idx, m_idx) = max(C_env(:, 2));
            
            time_loc(snr_idx, m_idx) = t_elapsed(m_idx);
        end
    end
    
    patcorr_1(mc_idx, :, :) = p_corr_1_loc;
    patcorr_2(mc_idx, :, :) = p_corr_2_loc;
    filcorr_test_1(mc_idx, :, :) = f_test_1_loc;
    filcorr_test_2(mc_idx, :, :) = f_test_2_loc;
    time_comp(mc_idx, :, :) = time_loc;
end

%% Отрисовка результатов
get_mean = @(arr) squeeze(mean(arr, 1));
get_ci   = @(arr) squeeze(1.96 * std(arr, 0, 1) / sqrt(nMC));

mp1 = get_mean(patcorr_1); cp1 = get_ci(patcorr_1);
mp2 = get_mean(patcorr_2); cp2 = get_ci(patcorr_2);
mf1 = get_mean(filcorr_test_1); cf1 = get_ci(filcorr_test_1);
mf2 = get_mean(filcorr_test_2); cf2 = get_ci(filcorr_test_2);
mt  = get_mean(time_comp);

x = snr_range; 
figure('Position', [50 50 1600 800], 'Color', 'w'); 
colors = [0.8 0 0; 0 0.5 1; 0 0.7 0]; % Красный(eSPoC), Синий(mSPoC), Зеленый(PCA+SPoC)
titles = {'Recovery of Pattern 1', 'Recovery of Pattern 2', 'Computational Time', ...
          'Test Data: Power Corr (Net 1)', 'Test Data: Power Corr (Net 2)'};
y_data = {mp1, mp2, mt, mf1, mf2};
ci_data = {cp1, cp2, [], cf1, cf2};

for p = 1:5
    subplot(2, 3, p); hold on;
    for m = 1:nMethods
        y = y_data{p}(:,m)'; 
        if p ~= 3 
            ci = ci_data{p}(:,m)';
            fill([x fliplr(x)], [y-ci fliplr(y+ci)], colors(m,:), 'FaceAlpha', 0.15, 'EdgeColor', 'none', 'HandleVisibility', 'off');   
        end
        plot(x, y, '-o', 'Color', colors(m,:), 'LineWidth', 2, 'MarkerFaceColor', 'w', 'DisplayName', labels{m});
    end
    
    title(titles{p}, 'FontSize', 12);
    xlabel('EEG Target Source SNR'); 
    
    if p == 3
        ylabel('Time (seconds)');
        set(gca, 'XScale', 'log', 'YScale', 'log');
    else
        ylabel('Correlation (r)');
        ylim([0 1.05]); 
        set(gca, 'XScale', 'log');
    end
    grid on; 
    if p == 1, legend('Location', 'southeast'); end
end