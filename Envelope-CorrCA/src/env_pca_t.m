function [z_trials, X_epochs, X_covs] = env_pca_t(X, Fs, Wsize, Ssize, lambda)
    if nargin < 5
        lambda = 1e-5; 
    end
    [~, n_channels, n_trials] = size(X);
    
    for tr_idx=1:n_trials
        X(:,:,tr_idx) = X(:,:,tr_idx) - mean(X(:,:,tr_idx),1);
    end
    Xmean = mean(X,3);
    
    temp_epo = epoch_data(X(:,:,1), Fs, Wsize, Ssize);
    [~, ~, n_epochs] = size(temp_epo);
    X_epochs = zeros(size(temp_epo,1), n_channels, n_epochs, n_trials);
    
    for tr_idx=1:n_trials
        mX = X(:,:,tr_idx) - Xmean;
        mX = mX ./ sqrt(trace(cov(mX))); 
        X_epochs(:,:,:,tr_idx) = epoch_data(mX,Fs,Wsize,Ssize);
    end
    
    X_covs = zeros(n_channels, n_channels, n_epochs, n_trials);
    for j=1:n_trials
        for i=1:n_epochs
            C = cov(X_epochs(:,:,i,j));
            C_reg = C + lambda * (trace(C) / n_channels) * eye(n_channels);
            X_covs(:,:,i,j) = (C_reg + C_reg') / 2; 
        end
    end
        
    X_covs_flat = reshape(X_covs, n_channels, n_channels, n_epochs * n_trials);
    Cm = riemann_mean(X_covs_flat);
    X_covsT_flat = Tangent_space(X_covs_flat, Cm);
    D_vec = size(X_covsT_flat, 1); 
    
    X_covsT_reshaped = reshape(X_covsT_flat, D_vec, n_epochs, n_trials);
    X_avgT = mean(X_covsT_reshaped, 3); 
    mean_vec = mean(X_avgT, 2); 
    X_avgT_centered = X_avgT - mean_vec;
    
    [U, ~, ~] = svd(X_avgT_centered, 'econ');
    X_covsT = permute(X_covsT_reshaped, [2, 1, 3]);
    
    n_comps = size(U, 2);
    z_trials = zeros(n_epochs, n_comps, n_trials);
    
    mean_vec_transposed = mean_vec';
    
    for j = 1:n_trials
        trial_data = X_covsT(:,:,j);
        trial_data_centered = trial_data - mean_vec_transposed;
        z_tr = trial_data_centered * U;
        z_trials(:,:,j) = (z_tr - mean(z_tr, 1)) ./ std(z_tr, [], 1);
    end
end

