close all
clear
clc

%% =================== ПАРАМЕТРЫ СИМУЛЯЦИИ ===================
Nsrc = 50;
Ndistr = 2;
flanker = 1;
Ts = 250; % Короткая запись (вычислительная эффективность на малых данных)
Ntr = 100;
Fs = 250;
Ws = 1;
Ss = 1;
nMC = 20;
Nsens = 64; % Мок для 64 каналов
Nsites = 2000; % Мок для случайной геометрии
fixed_eeg_snr = 10^0.4;
noise_range = logspace(-1, 2, 8);
nNoise = length(noise_range);

labels = {'eSPoC', 'mSPoC'};
nMethods = length(labels);

% Массивы для метрик
time_comp = zeros(nMC, nNoise, nMethods);
patcorr_1 = zeros(nMC, nNoise, nMethods);

% Мокируем матрицу лидфилда (G) для избежания внешних данных
G = randn(Nsens, Nsites * 3);

for mc_idx = 1:nMC
    fprintf('Monte-Carlo iteration: %d / %d\n', mc_idx, nMC);

    % Используем встроенную функцию репозитория (которая ожидает G)
    [X_s, X_bg, X_n, z, GA, S] = generate_distributed_sources(G, Nsrc, Ndistr, flanker, Ts, Fs);
    Ainit = GA(:, 1:2);

    X = fixed_eeg_snr * X_s + X_bg + 0.1 * X_n / norm(X_s,'fro');

    % Мокируем epoch_data если нужно, или используем из TriCo/code/core
    X_epo = epoch_data(X', Fs, Ws, Ss);
    X_epo_train = X_epo(:,:,1:Ntr);
    nTrain = size(X_epo_train, 3);

    z_epo_raw = epoch_data(z(1:2,:)', Fs, Ws, Ss);
    z_epo = squeeze(mean(z_epo_raw, 1))';
    z_train = z_epo(1:Ntr, :);
    z_train = (z_train - mean(z_train, 1)) ./ std(z_train, 0, 1);

    n_ext = 5;
    mix_mat = randn(n_ext, n_ext);

    time_loc = zeros(nNoise, nMethods);
    p_corr_1_loc = zeros(nNoise, nMethods);

    parfor noise_idx = 1:nNoise
        current_noise = noise_range(noise_idx);

        Z_multi_train = zeros(nTrain, n_ext);
        Z_multi_train(:, 1:2) = z_train;

        for d = 3:n_ext
            nd = randn(nTrain, 1);
            nd = nd - z_train(:,1)*(z_train(:,1)'*nd)/(z_train(:,1)'*z_train(:,1)) ...
                    - z_train(:,2)*(z_train(:,2)'*nd)/(z_train(:,2)'*z_train(:,2));
            Z_multi_train(:, d) = (nd - mean(nd)) / std(nd) * current_noise;
        end
        Z_noisy_train_multi = (mix_mat * Z_multi_train')';

        A_est = cell(1, nMethods);
        t_elapsed = zeros(1, nMethods);

        % eSPoC
        tic;
        [W_e, A_e] = espoc(X_epo_train, Z_noisy_train_multi');
        A_est{1} = squeeze(A_e(1, :, 1:2));
        t_elapsed(1) = toc;

        % mSPoC
        tic;
        mspoc_opts = struct('tau_vector', 0, 'n_component_sets', 2, 'n_random_initializations', 5, 'verbose', 0);
        [W_m, ~, ~, A_m, ~] = mspoc(X_epo_train, Z_noisy_train_multi', mspoc_opts);
        t_elapsed(2) = toc;
        A_est{2} = A_m(:, 1:2);

        for m_idx = 1:nMethods
            C_A = abs(corr(A_est{m_idx}, Ainit));
            p_corr_1_loc(noise_idx, m_idx) = max(C_A(:, 1));
            time_loc(noise_idx, m_idx) = t_elapsed(m_idx);
        end
    end

    patcorr_1(mc_idx, :, :) = p_corr_1_loc;
    time_comp(mc_idx, :, :) = time_loc;
end

%% Отрисовка
mp1 = squeeze(mean(patcorr_1, 1));
mt  = squeeze(mean(time_comp, 1));
cp1 = squeeze(1.96 * std(patcorr_1, 0, 1) / sqrt(nMC));

x = noise_range;
fig = figure('Position', [50 50 1200 500], 'Color', 'w', 'Visible', 'off');
colors = [0.8 0 0; 0 0.5 1];

subplot(1, 2, 1); hold on;
for m = 1:nMethods
    y = mp1(:,m)'; ci = cp1(:,m)';
    fill([x fliplr(x)], [y-ci fliplr(y+ci)], colors(m,:), 'FaceAlpha', 0.15, 'EdgeColor', 'none', 'HandleVisibility', 'off');
    plot(x, y, '-o', 'Color', colors(m,:), 'LineWidth', 2, 'MarkerFaceColor', 'w', 'DisplayName', labels{m});
end
title('Recovery of Pattern 1 (Small Data MOCK)', 'FontSize', 12);
xlabel('Orthogonal Noise Variance'); ylabel('Correlation (r)');
ylim([0 1.05]); set(gca, 'XScale', 'log'); grid on; legend('Location', 'southwest');

subplot(1, 2, 2); hold on;
for m = 1:nMethods
    plot(x, mt(:,m)', '-o', 'Color', colors(m,:), 'LineWidth', 2, 'MarkerFaceColor', 'w', 'DisplayName', labels{m});
end
title('Computational Time (Small Data MOCK)', 'FontSize', 12);
xlabel('Orthogonal Noise Variance'); ylabel('Time (seconds)');
set(gca, 'XScale', 'log', 'YScale', 'log'); grid on; legend('Location', 'northwest');

saveas(fig, 'simulation_efficiency_comparison.png');
disp('Simulation complete. Graph saved as simulation_efficiency_comparison.png');
