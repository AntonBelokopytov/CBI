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
Nsrc = 101;     
Ndistr = 2;     % Количество целевых нейрональных источников
Nmix = 2;       % Количество внешних (поведенческих) переменных
noise_level = 0.1; 
flanker = 1;
Ts = 850;      
Fs = 250;
Ws = 1;
Ss = 1;
nMC = 5;        
n_train_epochs = 250; 

SNR_range = 10.^(-1.4:0.2:1); 
nSNR = length(SNR_range);
labels = {'eSPoC', 'mSPoC'};
nMethods = length(labels);

% Массивы для метрик: (1) ЭЭГ Мощность, (2) Простр. Паттерны, (3) Раскрутка Внешней Переменной
filcorr_test_1 = zeros(nMC, nSNR, nMethods); 
filcorr_test_2 = zeros(nMC, nSNR, nMethods); 
patcorr_1      = zeros(nMC, nSNR, nMethods); 
patcorr_2      = zeros(nMC, nSNR, nMethods); 
zcorr_test_1   = zeros(nMC, nSNR, nMethods); % НОВОЕ: корреляция восстановленного Z
zcorr_test_2   = zeros(nMC, nSNR, nMethods); % НОВОЕ: корреляция восстановленного Z

for mc_idx = 1:nMC
    fprintf('Monte-Carlo iteration: %d / %d\n', mc_idx, nMC);
    
    % 1. Генерируем "чистые" источники
    [X_s, X_bg, X_n, z, GA, S] = generate_distributed_sources(G, Nsrc, Ndistr, flanker, Ts, Fs);
    
    Ainit = GA(:, 1:Ndistr); 
    
    z_epo_raw = epoch_data(z(1:Ndistr,:)', Fs, Ws, Ss); 
    z_epo = squeeze(mean(z_epo_raw, 1)); 
    
    for src_i = 1:Ndistr
        z_epo(src_i,:) = (z_epo(src_i,:) - mean(z_epo(src_i,:))) / std(z_epo(src_i,:));
    end
    
    z_train = z_epo(:, 1:n_train_epochs);
    z_test  = z_epo(:, n_train_epochs+1 : end);
    
    % ================= СИМУЛЯЦИЯ ВНЕШНИХ СЕНСОРОВ =================
    ext_weights = randn(Nmix, Ndistr); 
    
    z_multidim = ext_weights * z_epo + noise_level * randn(Nmix, size(z_epo, 2));
    z_multidim_train = z_multidim(:, 1:n_train_epochs);
    z_multidim_test  = z_multidim(:, n_train_epochs+1 : end); % НОВОЕ: Тестовая часть внешних сенсоров
    
    % Временные матрицы для parfor
    f_test_1_local = zeros(nSNR, nMethods); 
    f_test_2_local = zeros(nSNR, nMethods); 
    p_corr_1_local = zeros(nSNR, nMethods);
    p_corr_2_local = zeros(nSNR, nMethods);
    z_corr_1_local = zeros(nSNR, nMethods);
    z_corr_2_local = zeros(nSNR, nMethods);
    
    % 2. Перебираем разные уровни SNR
    parfor snr_idx = 1:nSNR
        current_SNR = SNR_range(snr_idx);
        
        X = current_SNR * X_s + X_bg + 0.1 * X_n / norm(X_s,'fro');
        X_epo = epoch_data(X', Fs, Ws, Ss);
        
        X_epo_train = X_epo(:,:, 1:n_train_epochs);
        X_epo_test  = X_epo(:,:, n_train_epochs+1 : end);
        
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
        
        % ================= Оценка методов =================
        w_all  = zeros(nChan, Ndistr, nMethods);
        a_all  = zeros(nChan, Ndistr, nMethods);
        vz_all = zeros(Nmix, Ndistr, nMethods); % НОВОЕ: Фильтры для поведенческих переменных
        
        % 1. eSPoC
        [W_e, A_e, ~, Vz, corrs_e, ~, cca_corrs] = espoc(X_epo_train, z_multidim_train);
        
        for f_idx = 1:Ndistr
            [~, idx] = sort(abs(corrs_e(f_idx,:)), 'descend');
            w_all(:, f_idx, 1)  = squeeze(W_e(f_idx, :, idx(1)))'; 
            a_all(:, f_idx, 1)  = squeeze(A_e(f_idx, :, idx(1)))';
            vz_all(:, f_idx, 1) = Vz(:, f_idx); % Извлекаем фильтр z для текущей компоненты
        end
        
        % 2. mSPoC
        mspoc_opts = struct('tau_vector', 0, 'n_component_sets', 2, 'verbose', 0);
        [W_m, Wy, ~, A_m, ~] = mspoc(X_epo_train, z_multidim_train, mspoc_opts);
        
        for f_idx = 1:Ndistr
            w_all(:, f_idx, 2)  = W_m(:, f_idx); 
            a_all(:, f_idx, 2)  = A_m(:, f_idx); 
            vz_all(:, f_idx, 2) = Wy(:, f_idx); % Извлекаем фильтр z (в mSPoC он называется Wy)
        end
        
        % ================= Проверка на тесте и Матчинг =================
        for m_idx = 1:nMethods
            w  = w_all(:,:,m_idx);
            a  = a_all(:,:,m_idx);
            vz = vz_all(:,:,m_idx);
            
            % Расчет тестовой огибающей ЭЭГ
            env_test = zeros(2, nTest);
            for c = 1:2
                for ep_idx = 1:nTest
                    env_test(c, ep_idx) = w(:,c)' * Covs_test(:,:,ep_idx) * w(:,c);
                end
            end
            
            % НОВОЕ: Раскрутка поведенческой смеси на тестовых данных
            z_rec_test = vz' * z_multidim_test; % Размерность [2 x nTest]
            
            % Считаем матрицу кросс-корреляций 2x2 (по мощности ЭЭГ) для матчинга
            corr_mat = abs(corr(env_test', z_test')); 
            
            sum_diag = corr_mat(1,1) + corr_mat(2,2);
            sum_anti = corr_mat(1,2) + corr_mat(2,1);
            
            if sum_diag >= sum_anti
                % Прямое совпадение: комп.1 -> Ист.1, комп.2 -> Ист.2
                f_test_1_local(snr_idx, m_idx) = corr_mat(1,1);
                f_test_2_local(snr_idx, m_idx) = corr_mat(2,2);
                p_corr_1_local(snr_idx, m_idx) = abs(corr(a(:,1), Ainit(:,1)));
                p_corr_2_local(snr_idx, m_idx) = abs(corr(a(:,2), Ainit(:,2)));
                
                % Оценка поведенческой раскрутки (z_rec vs z_true)
                z_corr_1_local(snr_idx, m_idx) = abs(corr(z_rec_test(1,:)', z_test(1,:)'));
                z_corr_2_local(snr_idx, m_idx) = abs(corr(z_rec_test(2,:)', z_test(2,:)'));
            else
                % Обратное совпадение (перекрестное): комп.1 -> Ист.2, комп.2 -> Ист.1
                f_test_1_local(snr_idx, m_idx) = corr_mat(2,1);
                f_test_2_local(snr_idx, m_idx) = corr_mat(1,2);
                p_corr_1_local(snr_idx, m_idx) = abs(corr(a(:,2), Ainit(:,1)));
                p_corr_2_local(snr_idx, m_idx) = abs(corr(a(:,1), Ainit(:,2)));
                
                % Оценка поведенческой раскрутки (кросс-матчинг)
                z_corr_1_local(snr_idx, m_idx) = abs(corr(z_rec_test(2,:)', z_test(1,:)'));
                z_corr_2_local(snr_idx, m_idx) = abs(corr(z_rec_test(1,:)', z_test(2,:)'));
            end
        end
    end
    
    % Запись результатов итерации MC
    filcorr_test_1(mc_idx, :, :) = f_test_1_local;
    filcorr_test_2(mc_idx, :, :) = f_test_2_local;
    patcorr_1(mc_idx, :, :)      = p_corr_1_local;
    patcorr_2(mc_idx, :, :)      = p_corr_2_local;
    zcorr_test_1(mc_idx, :, :)   = z_corr_1_local;
    zcorr_test_2(mc_idx, :, :)   = z_corr_2_local;
end

%% ================= Вычисление статистики =================
mean_f_1 = squeeze(mean(filcorr_test_1, 1));
mean_f_2 = squeeze(mean(filcorr_test_2, 1));
mean_p_1 = squeeze(mean(patcorr_1, 1));  
mean_p_2 = squeeze(mean(patcorr_2, 1));
mean_z_1 = squeeze(mean(zcorr_test_1, 1));
mean_z_2 = squeeze(mean(zcorr_test_2, 1));

ci_f_1 = squeeze(1.96 * std(filcorr_test_1, 0, 1) / sqrt(nMC));
ci_f_2 = squeeze(1.96 * std(filcorr_test_2, 0, 1) / sqrt(nMC));
ci_p_1 = squeeze(1.96 * std(patcorr_1, 0, 1) / sqrt(nMC));
ci_p_2 = squeeze(1.96 * std(patcorr_2, 0, 1) / sqrt(nMC));
ci_z_1 = squeeze(1.96 * std(zcorr_test_1, 0, 1) / sqrt(nMC)); % НОВОЕ
ci_z_2 = squeeze(1.96 * std(zcorr_test_2, 0, 1) / sqrt(nMC)); % НОВОЕ

% ================= Визуализация (Сетка 2x3) =================
x = SNR_range; 
% Расширяем фигуру под 3-й столбец
figure('Position', [100 100 1600 800], 'Color', 'w'); 
colors = [0.8 0 0;    % Красный для eSPoC
          0 0 0.8];   % Синий для mSPoC
markers = {'o', 's'};

% --- ROW 1: Source 1 ---
% 1. Power Correlation
subplot(2,3,1); hold on; box on;
for m = 1:nMethods
    y = mean_f_1(:,m)'; ci = ci_f_1(:,m)';
    fill([x fliplr(x)], [y-ci fliplr(y+ci)], colors(m,:), 'FaceAlpha', 0.15, 'EdgeColor', 'none', 'HandleVisibility', 'off');   
    semilogx(x, y, ['-', markers{m}], 'Color', colors(m,:), 'LineWidth', 2, 'MarkerSize', 5, 'MarkerFaceColor', 'w', 'DisplayName', labels{m});
end
title('Src 1: Power Correlation (Test)', 'FontSize', 12, 'FontWeight', 'bold');
ylabel('Correlation, r', 'FontSize', 11);
ylim([0 1.05]); xlim([min(x) max(x)]);
set(gca, 'XScale', 'log', 'GridAlpha', 0.3, 'MinorGridAlpha', 0.4, 'TickDir', 'out'); grid on;
legend('Location', 'northwest', 'FontSize', 10);

% 2. Pattern Correlation
subplot(2,3,2); hold on; box on;
for m = 1:nMethods
    y = mean_p_1(:,m)'; ci = ci_p_1(:,m)';
    fill([x fliplr(x)], [y-ci fliplr(y+ci)], colors(m,:), 'FaceAlpha', 0.15, 'EdgeColor', 'none', 'HandleVisibility', 'off');   
    semilogx(x, y, ['-', markers{m}], 'Color', colors(m,:), 'LineWidth', 2, 'MarkerSize', 5, 'MarkerFaceColor', 'w', 'DisplayName', labels{m});
end
title('Src 1: Spatial Pattern Corr', 'FontSize', 12, 'FontWeight', 'bold');
ylim([0 1.05]); xlim([min(x) max(x)]);
set(gca, 'XScale', 'log', 'GridAlpha', 0.3, 'MinorGridAlpha', 0.4, 'TickDir', 'out'); grid on;

% 3. Behavioral Unmixing Correlation
subplot(2,3,3); hold on; box on;
for m = 1:nMethods
    y = mean_z_1(:,m)'; ci = ci_z_1(:,m)';
    fill([x fliplr(x)], [y-ci fliplr(y+ci)], colors(m,:), 'FaceAlpha', 0.15, 'EdgeColor', 'none', 'HandleVisibility', 'off');   
    semilogx(x, y, ['-', markers{m}], 'Color', colors(m,:), 'LineWidth', 2, 'MarkerSize', 5, 'MarkerFaceColor', 'w', 'DisplayName', labels{m});
end
title('Src 1: Behavioral Var Corr', 'FontSize', 12, 'FontWeight', 'bold');
ylim([0 1.05]); xlim([min(x) max(x)]);
set(gca, 'XScale', 'log', 'GridAlpha', 0.3, 'MinorGridAlpha', 0.4, 'TickDir', 'out'); grid on;

% --- ROW 2: Source 2 ---
% 1. Power Correlation
subplot(2,3,4); hold on; box on;
for m = 1:nMethods
    y = mean_f_2(:,m)'; ci = ci_f_2(:,m)';
    fill([x fliplr(x)], [y-ci fliplr(y+ci)], colors(m,:), 'FaceAlpha', 0.15, 'EdgeColor', 'none', 'HandleVisibility', 'off');   
    semilogx(x, y, ['-', markers{m}], 'Color', colors(m,:), 'LineWidth', 2, 'MarkerSize', 5, 'MarkerFaceColor', 'w', 'DisplayName', labels{m});
end
title('Src 2: Power Correlation (Test)', 'FontSize', 12, 'FontWeight', 'bold');
xlabel('Signal-to-Noise Ratio (SNR)', 'FontSize', 11);
ylabel('Correlation, r', 'FontSize', 11);
ylim([0 1.05]); xlim([min(x) max(x)]);
set(gca, 'XScale', 'log', 'GridAlpha', 0.3, 'MinorGridAlpha', 0.4, 'TickDir', 'out'); grid on;

% 2. Pattern Correlation
subplot(2,3,5); hold on; box on;
for m = 1:nMethods
    y = mean_p_2(:,m)'; ci = ci_p_2(:,m)';
    fill([x fliplr(x)], [y-ci fliplr(y+ci)], colors(m,:), 'FaceAlpha', 0.15, 'EdgeColor', 'none', 'HandleVisibility', 'off');   
    semilogx(x, y, ['-', markers{m}], 'Color', colors(m,:), 'LineWidth', 2, 'MarkerSize', 5, 'MarkerFaceColor', 'w', 'DisplayName', labels{m});
end
title('Src 2: Spatial Pattern Corr', 'FontSize', 12, 'FontWeight', 'bold');
xlabel('Signal-to-Noise Ratio (SNR)', 'FontSize', 11);
ylim([0 1.05]); xlim([min(x) max(x)]);
set(gca, 'XScale', 'log', 'GridAlpha', 0.3, 'MinorGridAlpha', 0.4, 'TickDir', 'out'); grid on;

% 3. Behavioral Unmixing Correlation
subplot(2,3,6); hold on; box on;
for m = 1:nMethods
    y = mean_z_2(:,m)'; ci = ci_z_2(:,m)';
    fill([x fliplr(x)], [y-ci fliplr(y+ci)], colors(m,:), 'FaceAlpha', 0.15, 'EdgeColor', 'none', 'HandleVisibility', 'off');   
    semilogx(x, y, ['-', markers{m}], 'Color', colors(m,:), 'LineWidth', 2, 'MarkerSize', 5, 'MarkerFaceColor', 'w', 'DisplayName', labels{m});
end
title('Src 2: Behavioral Var Corr', 'FontSize', 12, 'FontWeight', 'bold');
xlabel('Signal-to-Noise Ratio (SNR)', 'FontSize', 11);
ylim([0 1.05]); xlim([min(x) max(x)]);
set(gca, 'XScale', 'log', 'GridAlpha', 0.3, 'MinorGridAlpha', 0.4, 'TickDir', 'out'); grid on;