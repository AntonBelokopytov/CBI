function [W, A] = my_spoc(X_covs, z_comp, lambda)
    [n_ch, ~, n_ep, n_tr] = size(X_covs); 
    
    Cm = mean(X_covs(:,:,:),3);
    Cm_r = Cm + lambda*eye(size(Cm))*trace(Cm)/size(Cm,1);
    Wm = Cm_r^-0.5;
    
    Cxz = zeros(n_ch);
    for tr_i=1:n_tr
        X_covsTr = X_covs(:,:,:,tr_i);
        z_compTr = z_comp(:,tr_i);

        X_covsTr = (X_covsTr - mean(X_covsTr,3)) ./ std(X_covsTr(:));
        z_compTr = (z_compTr - mean(z_compTr,1)) ./ std(z_compTr(:));
        
        for ep_i=1:n_ep
            Cxz = Cxz + X_covsTr(:,:,ep_i) * z_compTr(ep_i);
        end
    end
    
    [w_tilde, s] = eig(Wm * Cxz * Wm'); 
    [s, idxs] = sort(diag(s), 'descend'); 
    w_tilde = w_tilde(:, idxs);
    
    W = Wm' * w_tilde; 
    
    for i = 1:size(W, 2)
        W(:, i) = W(:, i) / sqrt(W(:, i)' * Cm * W(:, i));
    end
    
    A = Cm * W; 
    
    for i = 1:size(A, 2)
        norm_A = norm(A(:, i));
        A(:, i) = A(:, i) / norm_A;
    end
end
