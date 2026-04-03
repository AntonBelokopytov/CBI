function [z_trials, Z_epochs, X_epochs] = env_pca_h(X, Fs, Wsize, Ssize, lambda)
    if nargin < 5
        lambda = 1e-5; 
    end
    [T, D, n_trials] = size(X);
    
    % 1. Вычитание ERP
    for tr_idx=1:n_trials
        X(:,:,tr_idx) = X(:,:,tr_idx) - mean(X(:,:,tr_idx),1);
    end
    Xmean = mean(X,3);
    
    % Нормировка
    for tr_idx=1:n_trials
        mX = X(:,:,tr_idx) - Xmean;
        mX = mX ./ sqrt(trace(cov(mX)));
        X(:,:,tr_idx) = mX;
    end
    
    % 2. Преобразование Гильберта и обрезка краев
    Xh_full = zeros(T, D, n_trials);
    for tr_idx=1:n_trials
        Xh_full(:,:,tr_idx) = hilbert(X(:,:,tr_idx));
    end
    
    pad_len = round(Fs/4);
    Xh = Xh_full(pad_len+1 : end-pad_len, :, :);
    [T_short, ~, ~] = size(Xh);
    
    % 3. Вычисление мгновенной кросс-мощности
    K = D * (D + 1) / 2;
    X_power = zeros(T_short, K, n_trials); 
    
    for tr_idx = 1:n_trials
        Y = Xh(:, :, tr_idx);
        idx = 1;
        for i = 1:D
            for j = i:D
                if i == j
                    X_power(:, idx, tr_idx) = Y(:, i) .* conj(Y(:, j)); % Авто-мощность
                else
                    X_power(:, idx, tr_idx) = sqrt(2) * Y(:, i) .* conj(Y(:, j)); % Кросс-мощность
                end
                idx = idx + 1;
            end
        end
    end
        
    X_power_avg = mean(X_power, 3);  
    mean_vec = mean(X_power_avg, 1); 
    X_power_avg_centered = X_power_avg - mean_vec;
    
    [~, ~, Vc] = svd(X_power_avg_centered, 'econ');
    
    n_comps = size(Vc, 2);
    z_trials = zeros(T_short, n_comps, n_trials);
    
    for j = 1:n_trials
        trial_power = X_power(:,:,j);
        trial_power_centered = trial_power - mean_vec;
        z_tr = abs(trial_power_centered * Vc);
        z_trials(:,:,j) = (z_tr - mean(z_tr, 1)) ./ std(z_tr, [], 1);
    end
    
    temp_epo = epoch_data(X(pad_len+1:end-pad_len,:,1), Fs, Wsize, Ssize);
    [W_samples, ~, n_epochs] = size(temp_epo);
    X_epochs = zeros(W_samples, D, n_epochs, n_trials);
    
    for tr_idx=1:n_trials
        X_epochs(:,:,:,tr_idx) = epoch_data(X(pad_len+1:end-pad_len,:,tr_idx), Fs, Wsize, Ssize);
    end
    
    Z_epochs = zeros(n_epochs, n_comps, n_trials);
    
    for j = 1:n_trials
        Z_epo_raw = epoch_data(z_trials(:,:,j), Fs, Wsize, Ssize);
        Z_epo_mean = mean(Z_epo_raw, 1);
        Z_epo_matrix = reshape(Z_epo_mean, n_comps, n_epochs)';
        Z_epochs(:,:,j) = (Z_epo_matrix - mean(Z_epo_matrix, 1)) ./ std(Z_epo_matrix, [], 1);
    end
end

