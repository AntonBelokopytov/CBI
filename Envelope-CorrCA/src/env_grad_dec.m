function [Z_trials, W, A, X_epochs, X_covs] = env_grad_dec(X, Fs, Wsize, Ssize, lambda)
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
    
    C_grad = zeros(n_channels, n_channels);    
    for tr = 1:n_trials
        for i = 1:n_epochs-1
            C1 = X_covs(:,:,i,tr);
            C2 = X_covs(:,:,i+1,tr);
            
            dist = distance_riemann(C1, C2);
            dC = (C2 - C1) / dist;                         
            C_grad = C_grad + dC;
        end
    end
    C_grad = C_grad / (n_trials * (n_epochs - 1));

    Cm = mean(X_covs(:,:,:),3);
    Cm_r = Cm + lambda*eye(size(Cm))*trace(Cm)/size(Cm,1);
    Wm = Cm_r^-0.5;

    % [W, S] = eig(Wm * C_grad * Wm');
    [W, S] = eig(C_grad);
    
    [~, indx] = sort(diag(S), 'ascend');
    W = W(:, indx);
    
    A = Cm * W / (W' * Cm * W);
    
    Z_trials = zeros(n_epochs, n_channels, n_trials);
    for tr = 1:n_trials
        for ep = 1:n_epochs
            C_ep = X_covs(:,:,ep,tr);
            Z_trials(ep, :, tr) = diag(W' * C_ep * W)';
        end
        Z_trials(:,:,tr) = (Z_trials(:,:,tr) - mean(Z_trials(:,:,tr), 1)) ./ std(Z_trials(:,:,tr), [], 1);
    end
end

