function [W, A, z_trials, X_epochs] = env_corrca_t(X, Fs, Wsize, Ssize, lambda)
    if nargin < 5
        lambda = 1e-3; 
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
    X_covs_reg = zeros(n_channels, n_channels, n_epochs, n_trials);
    for j=1:n_trials
        for i=1:n_epochs
            C = cov(X_epochs(:,:,i,j));
            X_covs(:,:,i,j) = C;
            
            C_reg = C + lambda * (trace(C) / n_channels) * eye(n_channels);
            X_covs_reg(:,:,i,j) = (C_reg + C_reg') / 2; 
        end
    end
        
    X_covs_flat = reshape(X_covs_reg, n_channels, n_channels, n_epochs * n_trials);
    Cm = riemann_mean(X_covs_flat);
    X_covsT_flat = Tangent_space(X_covs_flat, Cm);
    
    D_vec = size(X_covsT_flat, 1); 
    X_covsT_reshaped = reshape(X_covsT_flat, D_vec, n_epochs, n_trials);
    
    X_covsT = permute(X_covsT_reshaped, [2, 1, 3]);
    
    [Vc, ~, ~] = corrca(X_covsT,lambda);
    
    n_comps = size(Vc, 2);
    z_trials = zeros(n_epochs, n_comps, n_trials);
    
    for j = 1:n_trials
        trial_data = X_covsT(:,:,j);
        z_tr = trial_data * Vc;
        z_trials(:,:,j) = (z_tr - mean(z_tr, 1)) ./ std(z_tr, [], 1);
    end
        
    W = zeros(n_comps, n_channels, n_channels);
    A = zeros(n_comps, n_channels, n_channels);
    
    for comp_i = 1:10
        z_trials_comp = squeeze(z_trials(:, comp_i, :));
        [w, a] = my_spoc(X_covs, z_trials_comp, lambda);
        
        W(comp_i,:,:) = w;
        A(comp_i,:,:) = a;
    end
end
