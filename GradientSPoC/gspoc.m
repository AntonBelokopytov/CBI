function [W, A, S] = gspoc(X_epochs, z, lambda, k)
    if nargin < 3
        lambda = 1e-2; 
    end
    
    z = (z - mean(z)) ./ std(z);

    [~, n_channels, n_epochs] = size(X_epochs);
    
    if nargin < 4
        k = 15; 
    end

    X_covs = zeros(n_channels, n_channels, n_epochs);
    X_covs_reg = zeros(n_channels, n_channels, n_epochs);
    
    for ep_i = 1:n_epochs
        C = cov(X_epochs(:,:,ep_i));
        X_covs(:,:,ep_i) = C;
        
        C_reg = C + lambda * (trace(C) / n_channels) * eye(n_channels);
        X_covs_reg(:,:,ep_i) = (C_reg + C_reg') / 2; 
    end
    
    Cm = riemann_mean(X_covs_reg); 
    
    C_inv_half = zeros(n_channels, n_channels, n_epochs);
    for ep = 1:n_epochs
        C_inv_half(:,:,ep) = X_covs_reg(:,:,ep)^(-1/2); 
    end
    
    Dists = zeros(n_epochs, n_epochs);
    parfor ep_j = 1:n_epochs
        Cj_whiten = C_inv_half(:,:,ep_j); 
        dist_col = zeros(n_epochs, 1);
        
        for ep_i = (ep_j+1):n_epochs
            Ci = X_covs_reg(:,:,ep_i);
            M = Cj_whiten * Ci * Cj_whiten;
            M = (M + M') / 2; 
            
            e = eig(M);
            dist_col(ep_i) = sqrt(sum(log(e).^2));
        end
        Dists(:, ep_j) = dist_col;
    end
    
    Dists = Dists + Dists';
    % t = median(Dists(Dists > 0)); 
    % DistsR = exp(-Dists / t);

    k = min(k, n_epochs-1);
    KNN_mask = false(n_epochs, n_epochs);
    for ep_i = 1:n_epochs
        [~, sorted_idx] = sort(Dists(:, ep_i), 'ascend');
        neighbors = sorted_idx(2:k+1); 
        KNN_mask(neighbors, ep_i) = true;
    end
    KNN_mask = KNN_mask | KNN_mask';
    
    Cgrad = zeros(n_channels);
    for ep_j = 1:n_epochs-1
        Cj = X_covs(:,:,ep_j); 
        zj = z(ep_j);
        
        local_Cgrad = zeros(n_channels);
        
        for ep_i = (ep_j+1):n_epochs
            if ~KNN_mask(ep_i, ep_j)
                continue; 
            end
            
            Ci = X_covs(:,:,ep_i);
            zi = z(ep_i);
            d = Dists(ep_i, ep_j);

            if d > 0
                local_Cgrad = local_Cgrad + (zj - zi) * (Cj - Ci) / d;
            end
        end
        
        Cgrad = Cgrad + local_Cgrad;
    end
    
    [W, S_matrix] = eig(Cgrad,Cm);
    [S, idx] = sort(diag(S_matrix), 'descend'); 
    W = W(:, idx);
    
    for w_i = 1:size(W,2)
        w = W(:,w_i) / sqrt((W(:,w_i)' * Cm * W(:,w_i)));
        W(:,w_i) = w;
    end
    
    A = Cm * W / (W'* Cm * W); 
end