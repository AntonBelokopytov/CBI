function [W_out, A_out, corrs, Epochs_cov, z] = env_laplace_dec2(X, Fs, Wsize, Ssize, N_neigb, lambda, n_plot_comps)    
    if nargin < 5 || isempty(N_neigb), N_neigb = []; end
    if nargin < 6 || isempty(lambda), lambda = 1e-6; end
    if nargin < 7 || isempty(n_plot_comps), n_plot_comps = 3; end
    
    [~, n_ch, n_trials] = size(X);
    
    for tr_idx=1:n_trials
        X(:,:,tr_idx) = X(:,:,tr_idx) - mean(X(:,:,tr_idx),1);
    end
    Xmean = mean(X,3);
    
    X_epochs = [];
    for tr_idx=1:n_trials
        mX = X(:,:,tr_idx) - Xmean;
        mX = mX ./ sqrt(trace(cov(mX)));
        X_epochs(:,:,:,tr_idx) = epoch_data(mX,Fs,Wsize,Ssize);
    end
    
    [~, ~, n_epochs, ~] = size(X_epochs);
    if isempty(N_neigb), N_neigb = n_epochs; end
    Epochs_cov = zeros(n_ch, n_ch, n_epochs, n_trials); 
    
    % 1. Считаем регуляризованные ковариации
    for i=1:n_epochs
        for j=1:n_trials
            C = cov(X_epochs(:,:,i,j));
            C_reg = C + lambda * (trace(C) / n_ch) * eye(n_ch);
            C_reg = (C_reg + C_reg') / 2; 
            Epochs_cov(:,:,i,j) = C_reg;
        end
    end
    
    % 2. Построение консенсусного графа Лапласа
    All_W = zeros(n_epochs, n_epochs, n_trials);
    for tr_idx=1:n_trials 
        Trial_Dists = calc_riemann_dists(Epochs_cov(:,:,:,tr_idx));
        All_W(:,:,tr_idx) = build_graph_from_dists(Trial_Dists, N_neigb);
    end
    
    W_graph = mean(All_W, 3);
    D_graph = diag(sum(W_graph, 2)); 
    L = D_graph - W_graph;
    
    % 3. Совместная диагонализация (поиск огибающих z)
    [V, S] = eig(L, D_graph);
    S = diag(S); 
    [S, idx] = sort(S,'ascend'); 
    V = V(:,idx);
    
    valid_idx = S > 0; 
    valid_idx(1) = false; % Убираем тривиальный вектор
    
    V = V(:,valid_idx);
    z = V; 
    z = (z - mean(z,1)) ./ std(z,[],1);
    
    % =====================================================================
    % 4. Восстановление пространственных фильтров и паттернов
    % =====================================================================
    
    % Вычисление средней ковариации и матрицы отбеливания (Whitening)
    Cm = mean(mean(Epochs_cov, 4), 3);
    gamma_w = 1e-5;
    Cm_r = Cm + gamma_w * eye(size(Cm)) * (trace(Cm) / size(Cm,1));
    Wm = Cm_r^(-0.5);
    
    % Векторизация отбеленных ковариаций (переход в касательное пространство)
    n_features = n_ch * (n_ch + 1) / 2;
    X_covsVecW = zeros(n_epochs, n_features, n_trials);
    for i=1:n_epochs
        for j=1:n_trials
            Cw = Wm * Epochs_cov(:,:,i,j) * Wm';
            X_covsVecW(i,:,j) = cov2upper(Cw);
        end
    end
    
    % Регрессия пространственных признаков на временную динамику z
    Mean_CovsVecW = mean(X_covsVecW, 3);
    Af = Mean_CovsVecW' * z; 
    
    n_z = size(z, 2);
    W_out = zeros(n_z, n_ch, n_ch); 
    A_out = zeros(n_z, n_ch, n_ch);
    eigenvals = zeros(n_z, n_ch);
    
    for i=1:n_z
        [w, a, s] = project_filters_to_manifold(Af(:,i), Wm, Cm);
        W_out(i,:,:) = w;
        A_out(i,:,:) = a;
        eigenvals(i,:) = s';
    end
    
    % 5. Вычисление корреляций между триалами
    corrs = intertr_corrs(W_out, Epochs_cov, n_plot_comps);
    
    % 6. Визуализация результатов
    visualize(z, eigenvals, corrs, n_plot_comps, Wsize, Ssize);
end

% =========================================================================
% Вспомогательные функции
% =========================================================================

function X_epo = epoch_data(X, Fs, Ws, Ss)
    W = fix(Ws*Fs);
    S = fix(Ss*Fs);
    range = 1:W; ep = 1;
    X_epo = [];
    while range(end) <= size(X,1)
        X_epo(:,:,ep) = X(range,:); 
        range = range + S; ep = ep + 1;
    end
end

function Dists = calc_riemann_dists(Covs)
    n = size(Covs,3);
    Dists = zeros(n);
    for i=1:n-1
        for j=i+1:n
            A = Covs(:,:,i);
            B = Covs(:,:,j);
            d = distance_riemann(A,B); 
            Dists(i,j) = d;
        end
    end
    Dists = (Dists + Dists');
end

function W = build_graph_from_dists(Dists, N_neigb)
    n = size(Dists, 1);
    k = max(10, min(N_neigb, n - 1)); 
    
    % 1. Находим локальный масштаб sigma для каждой эпохи
    % sigma_i - это расстояние до k-го соседа
    sigmas = zeros(n, 1);
    for i = 1:n
        distances_i = Dists(i, :);
        % Сортируем расстояния по возрастанию
        sorted_dists = sort(distances_i, 'ascend');
        
        % Теперь вызов (k + 1) абсолютно безопасен
        sigmas(i) = sorted_dists(k + 1);
        
        % Защита от нулевого sigma (если есть идентичные эпохи)
        if sigmas(i) < 1e-10
            sigmas(i) = 1e-10;
        end
    end
    
    % 2. Строим матрицу весов (Self-Tuning Gaussian Kernel)
    W = zeros(n, n);
    for i = 1:n
        for j = i+1:n
            % Возводим расстояние в квадрат для Гауссиана
            d_sq = Dists(i, j)^2; 
            
            % Перемножаем локальные масштабы точек i и j
            scale = sigmas(i) * sigmas(j); 
            
            % Вычисляем вес связи
            w_val = exp(-d_sq / scale);
            
            W(i, j) = w_val;
            W(j, i) = w_val; % Граф сразу получается симметричным
        end
    end
    
    W = W - diag(diag(W)); 
end

function a = distance_riemann(A,B)
    a = sqrt(sum(real(log(eig(A,B))).^2));
end

function [v] = cov2upper(C)
    upper_triu_mask = triu(true(size(C)),1);
    upper_mask = triu(true(size(C)));
    C(upper_triu_mask) = C(upper_triu_mask)*sqrt(2);
    upper_triangle = C(upper_mask);
    v = upper_triangle(:);
end

function C = upper2cov(v)
    n = (-1 + sqrt(1 + 8 * numel(v))) / 2;
    assert(mod(n,1) == 0, 'Vector length does not correspond to a triangular matrix.');
    C = zeros(n);
    upper_mask = triu(true(n));
    C(upper_mask) = v;
    upper_triu_mask = triu(true(n), 1);
    C(upper_triu_mask) = C(upper_triu_mask) / sqrt(2);
    C = C + triu(C, 1)';
end

function [W_pr, A_pr, S_pr] = project_filters_to_manifold(V, Wm, Cxx)
    WW = upper2cov(V);
    [Uw, S_mat] = eig(WW);
    S_pr = diag(S_mat);
    [S_pr, idxs] = sort(S_pr, 'descend');
    Uw = Uw(:, idxs);
    
    n_ch = size(Uw, 2);
    W_pr = zeros(n_ch, n_ch);
    A_pr = zeros(n_ch, n_ch);
    
    for local_src_idx = 1:n_ch
        wi = Wm * Uw(:, local_src_idx);
        Wprn = wi / sqrt(wi' * Cxx * wi);
        W_pr(:, local_src_idx) = Wprn;
        A_pr(:, local_src_idx) = Cxx * Wprn / (Wprn' * Cxx * Wprn);
    end
end

function corrs = intertr_corrs(W, X_covs, n_filters_to_eval)    
    [n_filters, ~, n_components] = size(W);
    [~, ~, n_epochs, n_trials] = size(X_covs);
    
    n_to_do = min(n_filters_to_eval, n_filters);
    corrs = zeros(n_to_do, n_components);
    
    for f_idx = 1:n_to_do
        for comp_idx = 1:n_components
            Envs = zeros(n_epochs, n_trials);
            w = squeeze(W(f_idx, :, comp_idx)); 
            if isrow(w), w = w'; end 
            
            for ep_idx = 1:n_epochs
                for tr_idx = 1:n_trials
                    Envs(ep_idx, tr_idx) = w' * X_covs(:, :, ep_idx, tr_idx) * w;
                end
            end
            inters_c = corr(Envs);
            corr_mask = triu(true(size(inters_c)), 1);
            corrs(f_idx, comp_idx) = mean(inters_c(corr_mask));
        end
    end
end

function visualize(z, eigenvalues, corrs, n_comp, Wsize, Ssize)
    [n_epochs, n_z] = size(z);
    [n_filters, n_ch] = size(eigenvalues); 
    n_plot = min(n_comp, n_z); 
    
    if n_plot > 0
        figure('Name', 'Env-Laplace Analysis', 'Color', 'w', ...
               'Position', [100, 100, 1100, 200 * n_plot + 100]); 
        
        t_z = (0:n_epochs-1) * Ssize + (Wsize / 2);
        
        for i = 1:n_plot
            subplot(n_plot, 2, 2*i - 1);
            plot(t_z, z(:, i), 'LineWidth', 1.3, 'Color', [0.2 0.4 0.8]);
            ylabel(sprintf('z_{%d}', i)); 
            xlim([t_z(1), t_z(end)]);
            grid on;
            
            if i == 1, title('Component Envelopes'); end
            if i == n_plot
                xlabel('Time (s)');
            else
                set(gca, 'XTickLabel', []); 
            end
            
            subplot(n_plot, 2, 2*i);
            yyaxis left
            stem(1:n_ch, eigenvalues(i, :), 'filled', 'MarkerSize', 3.5, 'Color', [0.1 0.1 0.7]);
            ylabel('Eig');
            set(gca, 'YColor', [0.1 0.1 0.7]);
            
            yyaxis right
            plot(1:n_ch, corrs(i, :), '-o', 'LineWidth', 1.1, 'MarkerSize', 4, 'Color', [0.7 0.1 0.1]);
            ylabel('Corr');
            set(gca, 'YColor', [0.7 0.1 0.1]);
            ylim([min(0, min(corrs(i,:))), 1.05]); 
            
            xlim([0.5, n_ch + 0.5]);
            grid on;
            
            if i == 1, title('Eigvals & Inter-trial Corrs'); end
            if i == n_plot
                xlabel('Component Index');
            else
                set(gca, 'XTickLabel', []);
            end
        end
    end
end
