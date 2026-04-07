function [W, A, z_trials, X_epochs, raw_var] = env_corrca(X, Fs, Wsize, Ssize, lambda)
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
    Wm = Cm_r^-0.5;

    D_vec = n_channels * (n_channels + 1) / 2;
    X_covsVecW = zeros(n_epochs, D_vec, n_trials);
    for j=1:n_trials
        for i=1:n_epochs
            X_covsVecW(i,:,j) = cov2upper(Wm * X_covs(:,:,i,j) * Wm')';
        end
    end
    
    [Vc, ~, ~] = corrca(X_covsVecW,lambda);
    
    n_comps = size(Vc, 2);
    z_trials = zeros(n_epochs, n_comps, n_trials);
    raw_var = zeros(n_comps, n_trials);
    
    for j = 1:n_trials
        trial_data = X_covsVecW(:,:,j);
        z_tr = trial_data * Vc; 
        
        raw_var(:, j) = var(z_tr, 0, 1)'; 
        
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

