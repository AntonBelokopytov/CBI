function [z] = env_laplace_dec(X, Fs, Wsize, Ssize, lambda, N_neigb)    
    % Входные параметры:
    % X       - данные [каналы x время x трайлы]
    % lambda  - регуляризация
    % N_neigb - количество ближайших соседей (k) для графа

    if nargin < 5 || isempty(lambda), lambda = 1e-10; end
    if nargin < 6 || isempty(N_neigb), N_neigb = 20; end
    
    [~, n_ch, n_trials] = size(X);
    
    for tr_idx=1:n_trials
        X(:,:,tr_idx) = X(:,:,tr_idx) - mean(X(:,:,tr_idx),1);
    end
    
    X_epochs = [];
    for tr_idx=1:n_trials
        mX = X(:,:,tr_idx);
        mX = mX ./ sqrt(trace(cov(mX)));
        X_epochs(:,:,:,tr_idx) = epoch_data(mX,Fs,Wsize,Ssize);
    end
    
    [~, ~, n_epochs, ~] = size(X_epochs);
    % Epochs_cov = zeros(n_ch, n_ch, n_epochs, n_trials); 
    Epochs_cov_reg = zeros(n_ch, n_ch, n_epochs, n_trials); 
    
    parfor i=1:n_epochs
        for j=1:n_trials
            C = cov(X_epochs(:,:,i,j));
            C_reg = C + lambda * (trace(C) / n_ch) * eye(n_ch);
            C_reg = (C_reg + C_reg') / 2; 
            % Epochs_cov(:,:,i,j) = C;
            Epochs_cov_reg(:,:,i,j) = C_reg;
        end
    end
    
    Dists = zeros(n_epochs, n_epochs, n_trials);
    parfor tr_idx=1:n_trials 
        Trial_Dists = calc_riemann_dists(Epochs_cov_reg(:,:,:,tr_idx));
        std_val = std(Trial_Dists(triu(true(n_epochs), 1)));
        Dists(:,:,tr_idx) = Trial_Dists ./ (std_val + eps);
    end
    
    SumDists = mean(Dists, 3);
    
    k_eff = min(N_neigb, n_epochs - 1);
    knn_mask = false(n_epochs);
    
    for i = 1:n_epochs
        [~, sort_idx] = sort(SumDists(i, :), 'ascend');
        neighbors = sort_idx(2:k_eff+1); 
        knn_mask(i, neighbors) = true;
    end
    
    knn_mask = knn_mask | knn_mask';
    
    % W = zeros(n_epochs);
    % W(knn_mask) = 1 ./ (SumDists(knn_mask).^2 + eps); 
    % W = W - diag(diag(W)); 
    distances_in_graph = SumDists(knn_mask);
    sigma = median(distances_in_graph); 
    
    % Для предотвращения деления на 0, если вдруг граф вырожден
    if sigma < eps
        sigma = 1; 
    end
    
    % 2. Применяем экспоненциальное ядро
    W = zeros(n_epochs);
    W(knn_mask) = exp( -(SumDists(knn_mask).^2) / (2 * sigma^2) );
    
    % 3. Обнуляем диагональ (расстояние от узла до самого себя равно 0, 
    % экспонента даст 1, но для Лапласиана самопетли не нужны)
    W = W - diag(diag(W));    
    
    deg = sum(W, 2);
    D_inv_sqrt = diag(1 ./ sqrt(deg + eps)); 
    
    L = eye(n_epochs) - D_inv_sqrt * W * D_inv_sqrt;
    L = (L + L') / 2; 
    
    [V, S] = eig(L);
    [S, idx] = sort(diag(S), 'ascend'); 
    V = V(:, idx);
    
    valid_idx = S > 1e-10; 
    V = V(:, valid_idx);
    
    z = D_inv_sqrt * V;     
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
            C1 = Covs(:,:,i);
            C2 = Covs(:,:,j);
            d = distance_riemann(C1,C2); 
            Dists(i,j) = d;
        end
    end
    Dists = (Dists + Dists');
end

function a = distance_riemann(A,B)
    a = sqrt(sum(log(eig(A,B)).^2));
end
