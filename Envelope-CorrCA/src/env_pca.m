function [z_trials, X_epochs, X_covs] = env_pca(X, Fs, Wsize, Ssize, lambda)
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
    
    [~, ~, n_epochs, ~] = size(X_epochs);
    X_covs = zeros(n_channels, n_channels, n_epochs, n_trials);
    for j=1:n_trials
        for i=1:n_epochs
            X_covs(:,:,i,j) = cov(X_epochs(:,:,i,j));
        end
    end

    Cm = mean(X_covs(:,:,:),3);
    Cm_r = Cm + lambda*eye(size(Cm))*trace(Cm)/size(Cm,1);
    Wm = real(Cm_r^-0.5);
    
    D_vec = n_channels * (n_channels + 1) / 2;
    X_covsVecW = zeros(D_vec, n_epochs, n_trials); 
    for j=1:n_trials
        for i=1:n_epochs
            X_covsVecW(:,i,j) = cov2upper(Wm * X_covs(:,:,i,j) * Wm');
        end
    end
        
    X_avg = mean(X_covsVecW, 3); 
    mean_vec = mean(X_avg, 2); 
    X_avg_centered = X_avg - mean_vec; 
    [U, ~, ~] = svd(X_avg_centered, 'econ'); 
    
    n_comps = size(U, 2);
    z_trials = zeros(n_epochs, n_comps, n_trials);
    
    for j = 1:n_trials
        trial_data = X_covsVecW(:,:,j);
        
        trial_data_centered = trial_data - mean_vec;
        
        z_tr = U' * trial_data_centered;
        z_tr = z_tr';
        z_trials(:,:,j) = (z_tr - mean(z_tr, 1)) ./ std(z_tr, [], 1);
    end
end
