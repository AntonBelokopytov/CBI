function [z] = env_laplace_dec2(X, Fs, Wsize, Ssize, N_neigb, lambda, n_plot_comps)    
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
    
    % Dists = zeros(n_epochs, n_epochs, n_trials);
    % parfor tr_idx=1:n_trials 
    %     Trial_Dists = calc_riemann_dists(Epochs_cov_reg(:,:,:,tr_idx));
    % 
    %     upper_tri_vals = Trial_Dists(triu(true(size(Trial_Dists)), 1));
    % 
    %     scale_factor = median(upper_tri_vals);
    %     if scale_factor == 0
    %         scale_factor = eps;
    %     end
    % 
    %     Trial_Dists = Trial_Dists ./ scale_factor;
    % 
    %     Dists(:,:,tr_idx) = Trial_Dists;
    % end    
    % mDists = mean(Dists,3);
    % W = build_w(mDists, n_epochs);
    % 
    % D_inv_sqrt = diag(1 ./ sqrt(sum(W, 2) + eps));
    % L_sym = eye(size(W)) - D_inv_sqrt * W * D_inv_sqrt;
    % L_sym = (L_sym + L_sym') / 2; 
    % 
    % [V, S] = eig(L_sym);
    % S = diag(S); 
    % [S, idx] = sort(S,'ascend'); 
    % V = V(:,idx);
    % 
    % V = V;
    % 
    % tol = 0; 
    % valid_idx = S > tol;
    % 
    % V = V(:,valid_idx);
    % z = V; 
    % z = (z - mean(z,1)) ./ std(z,[],1);
    % plot(z(:,2))

    L_stack = zeros(n_epochs, n_epochs, n_trials);
    parfor tr_idx=1:n_trials 
        Trial_Dists = calc_riemann_dists(Epochs_cov_reg(:,:,:,tr_idx));
        
        upper_tri_vals = Trial_Dists(triu(true(size(Trial_Dists)), 1));
        scale_factor = median(upper_tri_vals);
        if scale_factor == 0, scale_factor = eps; end
        Trial_Dists = Trial_Dists ./ scale_factor;
        
        % Строим граф и Лапласиан ИНДИВИДУАЛЬНО для трайла
        W_trial = build_w(Trial_Dists, n_epochs);
        
        D_inv_sqrt = diag(1 ./ sqrt(sum(W_trial, 2) + eps));
        L_sym = eye(n_epochs) - D_inv_sqrt * W_trial * D_inv_sqrt;
        L_sym = (L_sym + L_sym') / 2; % Принудительная симметрия
        
        L_stack(:,:,tr_idx) = L_sym;
    end    
    
    % 2. Совместная ортогональная диагонализация
    % Функция orthogonal_ajd ищет V, такую что V'*L_stack(:,:,k)*V ~ Диагональная
    V = orthogonal_ajd(L_stack); 
    
    % 3. Вычисление "средних" собственных значений для сортировки
    mean_eigenvals = zeros(n_epochs, 1);
    for k = 1:n_trials
        % Диагональ проекции лапласиана на общий базис
        lambda_k = diag(V' * L_stack(:,:,k) * V);
        mean_eigenvals = mean_eigenvals + lambda_k;
    end
    mean_eigenvals = mean_eigenvals / n_trials;
    
    % 4. Сортировка по возрастанию гладкости (начиная с самых низких частот)
    [S_mean, idx] = sort(mean_eigenvals, 'ascend');
    V = V(:, idx);
    
    % 5. Отсечение нулевой компоненты (которая константа для графа)
    tol = 1e-10; 
    valid_idx = S_mean > tol;
    
    V = V(:, valid_idx);
    
    % Если мы оставляем только запрошенное количество компонент:
    if size(V, 2) > n_plot_comps
        V = V(:, 1:n_plot_comps);
    end
    
    z = V; 
    z = (z - mean(z,1)) ./ std(z,[],1);
end

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
    eigenvals = real(eig(A,B));
    eigenvals(eigenvals <= 0) = eps; 
    a = sqrt(sum(log(eigenvals).^2));
end

function W = build_w(Dists,Nneig)    
    n = size(Dists,1);
    Nneig = min(n, fix(Nneig));
    W = zeros(n);
    for i=1:n
        [vals,idxs] = sort(Dists(i,:),'ascend');
        W(i,idxs(2:Nneig)) = 1 ./ (vals(2:Nneig) + eps);
    end
    W(logical(eye(size(W)))) = 0;
    
    W = max(W, W');
end

function V = orthogonal_ajd(C, tol, max_iter)
    % C: Трехмерная матрица (N x N x K), где K - количество матриц для совместной диагонализации
    % Выход: V - общая ортогональная матрица базисов
    
    if nargin < 2 || isempty(tol), tol = 1e-8; end
    if nargin < 3 || isempty(max_iter), max_iter = 100; end
    
    [n, ~, K] = size(C);
    V = eye(n);
    
    encore = true;
    iter = 0;
    
    while encore && (iter < max_iter)
        encore = false;
        iter = iter + 1;
        
        for p = 1:n-1
            for q = p+1:n
                % Извлекаем элементы p,q по всем K матрицам
                x = squeeze(C(p, p, :) - C(q, q, :));
                y = squeeze(C(p, q, :) + C(q, p, :)); 
                
                % Ищем оптимальный угол поворота (чтобы максимизировать дисперсию на диагонали)
                A11 = sum(x .* x);
                A12 = sum(x .* y);
                A22 = sum(y .* y);
                
                D = sqrt((A11 - A22)^2 + 4 * A12^2);
                if D < eps
                    continue;
                end
                
                if A11 > A22
                    c2 = A11 - A22 + D;
                    s2 = 2 * A12;
                else
                    c2 = 2 * A12;
                    s2 = A22 - A11 + D;
                end
                
                norm_u = sqrt(c2^2 + s2^2);
                if norm_u < eps
                    continue;
                end
                c2 = c2 / norm_u;
                s2 = s2 / norm_u;
                
                % Вычисляем косинус и синус угла
                theta = atan2(s2, c2) / 2;
                c = cos(theta);
                s = sin(theta);
                
                % Если угол значимый, применяем вращение ко всем матрицам
                if abs(s) > tol
                    encore = true;
                    
                    for k = 1:K
                        % Обновление столбцов
                        Cp = C(:, p, k);
                        Cq = C(:, q, k);
                        C(:, p, k) = c * Cp + s * Cq;
                        C(:, q, k) = -s * Cp + c * Cq;
                        
                        % Обновление строк
                        Row_p = C(p, :, k);
                        Row_q = C(q, :, k);
                        C(p, :, k) = c * Row_p + s * Row_q;
                        C(q, :, k) = -s * Row_p + c * Row_q;
                    end
                    
                    % Применяем то же вращение к общей матрице V
                    Vp = V(:, p);
                    Vq = V(:, q);
                    V(:, p) = c * Vp + s * Vq;
                    V(:, q) = -s * Vp + c * Vq;
                end
            end
        end
    end
end
