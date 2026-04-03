function [Z_trials, X_epochs, X_covs, V_common] = env_laplace_dec(X, Fs, Wsize, Ssize, lambda)
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
        
    L_trials = zeros(n_epochs, n_epochs, n_trials);
    
    for tr = 1:n_trials
        Covs_tr = X_covs(:,:,:,tr);
        
        Dists = compute_dists(Covs_tr);
        
        A = 1 ./ (Dists + eps); 
        
        A = A - diag(diag(A)); 
        
        deg = sum(A, 2);
        D_inv_sqrt = diag(1 ./ sqrt(deg + eps)); 
        L_norm = eye(n_epochs) - D_inv_sqrt * A * D_inv_sqrt;
        
        L_trials(:,:,tr) = L_norm;
    end
    
    L_comm = mean(L_trials, 3);
    
    [V, S] = eig(L_comm);
    [~, indx] = sort(diag(S), 'ascend');
    V = V(:, indx); V = V(:,2:end);
    
    n_comps = size(V,2); 
    Z_trials = zeros(n_epochs, n_comps, n_trials);
    for tr = 1:n_trials
        z_tr = L_trials(:,:,tr) * V;
        Z_trials(:,:,j) = (z_tr - mean(z_tr, 1)) ./ std(z_tr, [], 1);
    end
end

function Dists = compute_dists(Covs)
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
    Dists = Dists + Dists';
end