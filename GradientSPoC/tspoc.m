function [W, A, S] = tspoc(X_epochs, z, min_var_explained)
    if nargin < 3
        min_var_explained = 0.99; 
    end
    
    z = (z - mean(z)) ./ std(z);
    [n_samples, n_channels, n_epochs] = size(X_epochs);
    
    Xraw = reshape(permute(X_epochs, [1, 3, 2]), [n_samples * n_epochs, n_channels]);
    
    Xraw = Xraw - mean(Xraw, 1);
    
    [~, S_svd, V] = svd(Xraw, 'econ');
    
    ev_sorted = diag(S_svd).^2 / (size(Xraw, 1) - 1);
    
    tol = ev_sorted(1) * 10^-6;
    r = sum(ev_sorted > tol);
    
    var_explained = cumsum(ev_sorted) / sum(ev_sorted);
    var_explained(end) = 1;
    n_components = find(var_explained >= min_var_explained, 1);
    n_components = min(n_components, r);
    
    M = diag(ev_sorted(1:n_components).^(-0.5)) * V(:, 1:n_components)';
    
    X_covs_white = zeros(n_components, n_components, n_epochs);
    C_mean_arithmetic = zeros(n_channels, n_channels);
    
    for ep_i = 1:n_epochs
        C = cov(X_epochs(:,:,ep_i));
        C_mean_arithmetic = C_mean_arithmetic + C;
        X_covs_white(:,:,ep_i) = M * C * M';
    end
    C_mean_arithmetic = C_mean_arithmetic / n_epochs;
    
    Cm_white = riemann_mean(X_covs_white); 
    Wm_white = Cm_white^(-0.5);
    
    Tcovs = Tangent_space(X_covs_white, Cm_white);
    
    TcovsW = Tcovs .* reshape(z, 1, []); 
    
    TcovsWd = mean(TcovsW, 2); 
    TcovsWd = TcovsWd ./ norm(TcovsWd);
    
    Cvar_white = UnTangent_space(TcovsWd, Cm_white);
    
    CvarW_white = Wm_white * Cvar_white * Wm_white';
    CvarW_white = (CvarW_white + CvarW_white') / 2; 
    
    [W_tilde, S_matrix] = eig(CvarW_white);
    [S, idx] = sort(diag(S_matrix), 'descend');
    W_tilde = W_tilde(:, idx);
    
    W_white = Wm_white' * W_tilde;
    
    for w_i = 1:size(W_white,2)
        w = W_white(:,w_i) / sqrt((W_white(:,w_i)' * Cm_white * W_white(:,w_i)));
        W_white(:,w_i) = w;
    end
    
    W = M' * W_white; 
    
    A = C_mean_arithmetic * W / (W' * C_mean_arithmetic * W); 
end