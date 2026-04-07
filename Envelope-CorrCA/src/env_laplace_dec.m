function [W, A, z_trials, X_epochs, raw_var] = env_laplace_dec(X, Fs, Wsize, Ssize, lambda, k)
    if nargin < 5
        lambda = 1e-5; 
    end
    if nargin < 6
        k = 20;
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
    X_covs_r = zeros(n_channels, n_channels, n_epochs, n_trials);
    for j=1:n_trials
        for i=1:n_epochs
            C = cov(X_epochs(:,:,i,j));
            X_covs(:,:,i,j) = C;

            C_reg = C + lambda * (trace(C) / n_channels) * eye(n_channels);
            X_covs_r(:,:,i,j) = (C_reg + C_reg') / 2; 
        end
    end
        
    L_trials = zeros(n_epochs, n_epochs, n_trials);
    
    k = fix(n_epochs / 4);
    for tr = 1:n_trials
        Covs_tr = X_covs_r(:,:,:,tr);
        
        Dists = compute_dists(Covs_tr);
        
        k_eff = min(k, n_epochs - 1); 
        knn_mask = false(n_epochs);
        
        for i = 1:n_epochs
            [~, sort_idx] = sort(Dists(i, :), 'ascend');
            neighbors = sort_idx(2:k_eff+1);
            knn_mask(i, neighbors) = true;
        end
        
        knn_mask = knn_mask | knn_mask';
        
        edge_dists = Dists(knn_mask & triu(true(n_epochs), 1));
        if isempty(edge_dists)
            t = eps;
        else
            t = median(edge_dists.^2);
            if t == 0
                t = eps; 
            end
        end
        
        Adj = zeros(n_epochs);
        % Adj(knn_mask) = exp(-(Dists(knn_mask).^2) / t); 
        Adj(knn_mask) = 1 ./ Dists(knn_mask).^2; 
        
        Adj = Adj - diag(diag(Adj)); 
                
        deg = sum(Adj, 2);
        D_inv_sqrt = diag(1 ./ sqrt(deg + eps)); 
        L_norm = eye(n_epochs) - D_inv_sqrt * Adj * D_inv_sqrt;
        
        L_trials(:,:,tr) = L_norm;
    end
    
    L_comm = mean(L_trials, 3);
    L_comm = (L_comm + L_comm') / 2;
    
    [V, S] = eig(L_comm);
    [~, indx] = sort(diag(S), 'ascend');
    V = V(:, indx); 
    V = V(:, 2:end); 
    
    n_comps = size(V,2); 
    z_trials = zeros(n_epochs, n_comps, n_trials);
    raw_var = zeros(n_comps, n_trials); 
    
    for tr = 1:n_trials
        z_tr = L_trials(:,:,tr) * V;
        
        raw_var(:, tr) = var(z_tr, 0, 1)';
        z_trials(:,:,tr) = (z_tr - mean(z_tr, 1)) ./ std(z_tr, [], 1);
    end
    
    comps_to_spoc = min(10, n_comps);
    
    W = zeros(comps_to_spoc, n_channels, n_channels);
    A = zeros(comps_to_spoc, n_channels, n_channels);
    
    for comp_i = 1:comps_to_spoc
        z_trials_comp = squeeze(z_trials(:, comp_i, :));
        [w, a] = my_spoc(X_covs, z_trials_comp, lambda);
        
        W(comp_i, :, :) = w;
        A(comp_i, :, :) = a;
    end
end

function Dists = compute_dists(Covs)
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
    Dists = Dists + Dists';
end
