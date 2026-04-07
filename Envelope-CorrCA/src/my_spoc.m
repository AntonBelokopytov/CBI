function [W, A] = my_spoc(X_covs, z_comp, lambda)
    n_ch = size(X_covs, 1); 
    
    Cm = mean(X_covs, 3);
    Cm_reg = Cm + lambda * (trace(Cm) / n_ch) * eye(n_ch);
    
    Wm = Cm_reg^(-0.5); 
    
    Cxz = zeros(n_ch);
    for ep_i = 1:numel(z_comp)
        Cxz = Cxz + X_covs(:,:,ep_i) * z_comp(ep_i);
    end
    
    [w_tilde, s] = eig(Wm * Cxz * Wm'); 
    [s, idxs] = sort(diag(s), 'descend'); 
    w_tilde = w_tilde(:, idxs);
    
    W = Wm' * w_tilde; 
    
    for i = 1:size(W, 2)
        W(:, i) = W(:, i) / sqrt(W(:, i)' * Cm_reg * W(:, i));
    end
    
    A = Cm_reg * W; 
    
    for i = 1:size(A, 2)
        norm_A = norm(A(:, i));
        A(:, i) = A(:, i) / norm_A;
    end
end
