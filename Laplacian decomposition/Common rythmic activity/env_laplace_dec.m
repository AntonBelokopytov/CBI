function [A_out, W_out, z, Epochs_cov] = env_laplace_dec2(X, Fs, Wsize, Ssize, N_neigb, lambda, n_plot_comps)    
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
    Epochs_cov_reg = zeros(n_ch, n_ch, n_epochs, n_trials); 
    
    parfor i=1:n_epochs
        for j=1:n_trials
            C = cov(X_epochs(:,:,i,j));
            C_reg = C + lambda * (trace(C) / n_ch) * eye(n_ch);
            C_reg = (C_reg + C_reg') / 2; 

            Epochs_cov(:,:,i,j) = C;
            Epochs_cov_reg(:,:,i,j) = C_reg;
        end
    end
    
    Dists = zeros(n_epochs, n_epochs, n_trials);
    parfor tr_idx=1:n_trials 
        Trial_Dists = calc_riemann_dists(Epochs_cov_reg(:,:,:,tr_idx));
        Trial_Dists = Trial_Dists ./ std( Trial_Dists(triu( true(size(Trial_Dists,1)) ,1)) );
        Dists(:,:,tr_idx) = Trial_Dists;
    end
    
    mDists = mean(Dists,3);
    W = build_w(mDists);
    
    D = diag(sum(W, 2)); 
    L = D - W;
    
    [V, S] = eig(L, D);
    S = diag(S); 
    [S, idx] = sort(S,'ascend'); 
    V = V(:,idx);
    
    valid_idx = S > 0; 
    valid_idx(1) = false;
    
    V = V(:,valid_idx);
    z = V; 
    z = (z - mean(z,1)) ./ std(z,[],1);
    
    % =====================================================================
    % 4. Индивидуальный SPoC для каждого трайла по огибающей z
    % =====================================================================
    n_z = 3;
    
    % Теперь фильтры и паттерны хранятся для каждого трайла отдельно
    % Размерности: [Индекс_z, Индекс_компоненты_SPoC, Каналы, Трайлы]
    W_out = zeros(n_z, n_ch, n_ch, n_trials); 
    A_out = zeros(n_z, n_ch, n_ch, n_trials);
    eigenvals_all = zeros(n_z, n_ch, n_trials);
    
    parfor c = 1:n_z
        z_c = z(:, c); % Берем целевую огибающую (она уже с нулевым средним)
        
        for tr = 1:n_trials
            % Вытягиваем ковариации для данного трайла
            C_tr = squeeze(Epochs_cov(:,:,:,tr)); 
            
            % 1. Средняя ковариация (базовый уровень)
            C_mean = mean(C_tr, 3);
                        
            % 2. z-взвешенная ковариация (матрица комодуляции)
            C_tr = C_tr - C_mean;

            gamma = 0.00001;
            C_m_reg = C_mean+gamma*eye(size(C_mean))*trace(C_mean)/size(C_mean,1);
            Wm = C_m_reg^-0.5;

            C_z = zeros(n_ch, n_ch);
            for ep = 1:n_epochs
                C_z = C_z + Wm * C_tr(:,:,ep) * z_c(ep) * Wm';
            end
            C_z = C_z / n_epochs;
            
            % 3. Решаем GEVP: C_z * W = lambda * C_mean * W
            [W_spoc, D_spoc] = eig(C_z);
            evals = diag(D_spoc);
            
            % Сортируем по абсолютному значению (модулю комодуляции)
            [~, sort_idx] = sort(abs(evals), 'descend');
            W_spoc = Wm * W_spoc(:, sort_idx);
            evals = evals(sort_idx);
            
            % 4. Нормировка фильтров и вычисление паттернов
            A_spoc = zeros(n_ch, n_ch);
            for f_idx = 1:n_ch
                w_f = W_spoc(:, f_idx);
                % Нормируем фильтр, чтобы дисперсия была равна 1 (w' * C_mean * w = 1)
                w_norm = w_f / sqrt(w_f' * C_mean * w_f);
                W_spoc(:, f_idx) = w_norm;
                
                % Паттерн = C_mean * W
                A_spoc(:, f_idx) = C_mean * w_norm;
            end
            
            % Сохраняем фильтры (транспонируем для удобства применения)
            W_out(c, :, :, tr) = W_spoc'; 
            A_out(c, :, :, tr) = A_spoc;
            eigenvals_all(c, :, tr) = evals';
        end
    end
    
    % Для визуализатора усредним модули собственных значений
    eigenvals_mean = mean(abs(eigenvals_all), 3);
    
    % =====================================================================
    % 5. Вычисление корреляций между триалами
    % =====================================================================
    corrs = intertr_corrs(W_out, Epochs_cov_reg, n_plot_comps);
    
    % 6. Визуализация результатов
    visualize(z, eigenvals_mean, corrs, n_plot_comps, Wsize, Ssize);
    
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

function a = distance_riemann(A,B)

    a = sqrt(sum(log(eig(A,B)).^2));

end

function W = build_w(Dists)    
    % W = exp(-((Dists - mu).^2) ./ sigma);
    W = 1 ./ Dists; 
    W(logical(eye(size(W)))) = 0;
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
    % W имеет размер: [n_z, n_components, n_ch, n_trials]
    [n_z, n_components, n_ch, n_trials] = size(W);
    [~, ~, n_epochs, ~] = size(X_covs);
    
    n_to_do = min(n_filters_to_eval, n_z);
    corrs = zeros(n_to_do, n_components);
    
    for f_idx = 1:n_to_do
        for comp_idx = 1:n_components
            Envs = zeros(n_epochs, n_trials);
            
            % Применяем ИНДИВИДУАЛЬНЫЙ фильтр к каждому трайлу
            for tr_idx = 1:n_trials
                w = squeeze(W(f_idx, comp_idx, :, tr_idx)); 
                if isrow(w), w = w'; end 
                
                for ep_idx = 1:n_epochs
                    Envs(ep_idx, tr_idx) = w' * X_covs(:, :, ep_idx, tr_idx) * w;
                end
            end
            
            % Считаем попарную межтрайловую корреляцию полученных огибающих
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
