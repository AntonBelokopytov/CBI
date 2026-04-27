function [W, A, corrs_in, corrs_ex, corrs_in_ex, Zpr_in, Zpr_ex] = espoct_csp(X_epochs, Z, varargin)
% ESPOCT_CSP - Fully Riemannian eSPoC for spatial filtering.
    
    opt = propertylist2struct(varargin{:});
    opt = set_defaults(opt, ...
                      'X_min_var_explained', 1, ...
                      'whitening_reg', 10e-5, ...
                      'cca_mode', 'regularized', ...
                      'cca_reg', 10e-5);
                      
    % Step 1: Base Covariances
    [~, n_channels, n_epochs] = size(X_epochs);
    Epochs_cov = zeros(n_channels, n_channels, n_epochs);
    for i = 1:n_epochs
        Epochs_cov(:,:,i) = regularize(cov(X_epochs(:,:,i)));
    end
    Cxx = mean(Epochs_cov, 3);
    
    % Step 2: Map to Tangent Space to get features for CCA
    [Tcovs, S_epochs, C_ref_half] = get_tangent_features(Epochs_cov, Cxx);
    
    % Optional but recommended: PCA to avoid P > N issues in CCA
    [Tcovs_pca, U_pca] = project_to_pc(Tcovs, opt.X_min_var_explained);
    
    % Step 3: Denoise target via CCA in Tangent Space
    if strcmp(opt.cca_mode, 'regularized')
        [Vt_pca, Vz, corrs_in_ex] = cca(Tcovs_pca', Z', opt);
    elseif strcmp(opt.cca_mode, 'standard')
        [Vt_pca, Vz] = canoncorr(Tcovs_pca', Z');
    else
        error('Unknown cca_mode');
    end
    
    % Project weights back from PCA space to full tangent feature space
    Vt = U_pca * Vt_pca;
    Z_in = Vt' * Tcovs;
    
    n_global_src = size(Z_in, 1);
    
    % Step 4: Riemannian spatial filter synthesis
    for global_src_idx = 1:n_global_src        
        z_in_current = Z_in(global_src_idx, :);
        z_ex_current = Vz(:, global_src_idx)' * (Z(global_src_idx,:) - mean(Z(global_src_idx,:)));
        
        Zpr_in(global_src_idx, :) = z_in_current;
        Zpr_ex(global_src_idx, :) = z_ex_current;
                
        % Riemannian projection to extract clean patterns
        [w, a, s] = synthesize_rcsp(z_in_current, S_epochs, C_ref_half);
        
        % Evaluate performance
        Env = zeros(1, size(Epochs_cov, 3));
        for local_src_idx = 1:size(w, 2)
            for ep_idx = 1:size(Epochs_cov, 3)
                Env(ep_idx) = w(:, local_src_idx)' * Epochs_cov(:, :, ep_idx) * w(:, local_src_idx);
            end
            cr_in(local_src_idx) = corr(Env', Zpr_in(global_src_idx, :)');
            cr_ex(local_src_idx) = corr(Env', Zpr_ex(global_src_idx, :)');
        end
        
        eigenvalues(global_src_idx, :) = s;
        corrs_in(global_src_idx, :) = cr_in;
        corrs_ex(global_src_idx, :) = cr_ex;
        W(global_src_idx, :, :) = w;
        A(global_src_idx, :, :) = a;
    end
    
    if size(W, 1) == 1
        corrs_in = squeeze(corrs_in);
        corrs_ex = squeeze(corrs_ex);
        W = squeeze(W);
        A = squeeze(A);
    end
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% TANGENT SPACE PROJECTIONS
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [T_feat, S_epochs, C_ref_half] = get_tangent_features(Epochs_cov, Cxx)
    [n_chan, ~, n_epochs] = size(Epochs_cov);
    n_features = (n_chan^2 - n_chan) / 2 + n_chan;
    
    % Matrix roots for whitening based on reference point (Cxx)
    [V_ref, D_ref] = eig(Cxx);
    d_cxx = diag(D_ref);
    d_cxx(d_cxx < eps) = eps; 
    
    C_ref_inv_half = V_ref * diag(1 ./ sqrt(d_cxx)) * V_ref';
    C_ref_half     = V_ref * diag(sqrt(d_cxx)) * V_ref'; 
    
    C_ref_inv_half = real((C_ref_inv_half + C_ref_inv_half') / 2);
    C_ref_half     = real((C_ref_half + C_ref_half') / 2);
    
    S_epochs = zeros(n_chan, n_chan, n_epochs);
    T_feat = zeros(n_features, n_epochs);
    
    for i = 1:n_epochs
        C_i = (Epochs_cov(:, :, i) + Epochs_cov(:, :, i)') / 2; 
        
        % Whitening
        C_rel = C_ref_inv_half * C_i * C_ref_inv_half;
        C_rel = (C_rel + C_rel') / 2; 
        
        % Riemannian logarithm
        [V_rel, D_rel] = eig(C_rel);
        d_rel = diag(D_rel);
        d_rel(d_rel < eps) = eps;
        
        S_i = V_rel * diag(log(d_rel)) * V_rel';
        S_i = real((S_i + S_i') / 2); 
        
        S_epochs(:, :, i) = S_i;
        T_feat(:, i) = cov2upper(S_i);
    end
end

function [W, A, s] = synthesize_rcsp(Z_in, S_epochs, C_ref_half)    
    [n_chan, ~, n_epochs] = size(S_epochs);
    
    % 1. Split target into positive (activation) and negative (deactivation) poles
    Z_in = Z_in - median(Z_in);
    Z_pos = max(0, Z_in);  
    Z_neg = max(0, -Z_in); 
    
    sum_pos = max(sum(Z_pos), eps); 
    sum_neg = max(sum(Z_neg), eps);
    
    A_pos = zeros(n_chan, n_chan);
    A_neg = zeros(n_chan, n_chan);
    
    % 2. Compute weighted Riemannian means in tangent space
    for i = 1:n_epochs
        A_pos = A_pos + Z_pos(i) * S_epochs(:, :, i);
        A_neg = A_neg + Z_neg(i) * S_epochs(:, :, i);
    end
    
    A_pos = (A_pos + A_pos') / 2 / sum_pos;
    A_neg = (A_neg + A_neg') / 2 / sum_neg;
    
    % 3. Map back to manifold via Riemannian exponential
    C_pos = C_ref_half * expm(A_pos) * C_ref_half;
    C_pos = real((C_pos + C_pos') / 2);
        
    C_neg = C_ref_half * expm(A_neg) * C_ref_half;
    C_neg = real((C_neg + C_neg') / 2);
        
    [Uw, S] = eig(C_pos, C_neg);    
    [s, idxs] = sort(diag(S), 'descend');
    Uw = Uw(:, idxs);
    
    W = zeros(n_chan, n_chan);
    A = zeros(n_chan, n_chan);
    
    for local_src_idx = 1:n_chan
        wi = real(Uw(:, local_src_idx)); 
        
        % Safeguard normalization
        norm_val = wi' * C_neg * wi;
        if norm_val < eps, norm_val = eps; end
        
        Wprn = wi / sqrt(norm_val);
        W(:, local_src_idx) = Wprn;
        A(:, local_src_idx) = C_neg * Wprn; 
    end
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% HELPER FUNCTIONS
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [v] = cov2upper(C)
    upper_triu_mask = triu(true(size(C)), 1);
    upper_mask = triu(true(size(C)));
    C(upper_triu_mask) = C(upper_triu_mask) * sqrt(2);
    upper_triangle = C(upper_mask);
    v = upper_triangle(:);
end

function [X_proj, U] = project_to_pc(X, min_var_explained)
    X = X - mean(X, 2);
    [U, S, ~] = svd(X, "econ");
    S = diag(S);
    tol_rank = max(size(X)) * eps(S(1));
    r = sum(S > tol_rank);
    ve = S.^2;
    var_explained = cumsum(ve) / sum(ve);
    var_explained(end) = 1;
    tol = 1e-12;
    
    n_components = find(var_explained >= min_var_explained - tol, 1);
    if isempty(n_components)
        n_components = r;
    end
    U = U(:, 1:n_components);
    X_proj = U' * X;
end

function [Vx, Vy, S] = cca(X, Y, opt)
    gamma = opt.cca_reg;
    X = X - mean(X, 1);  
    Y = Y - mean(Y, 1);
    [n, ~] = size(X);
    
    Cxx = (X' * X) / (n - 1);
    Cyy = (Y' * Y) / (n - 1);
    Cxy = (X' * Y) / (n - 1);
    
    scale_x = trace(Cxx) / size(Cxx, 1);
    scale_y = trace(Cyy) / size(Cyy, 1);
    
    Sxx_r = (1 - gamma) * Cxx + gamma * scale_x * eye(size(Cxx));
    Syy_r = (1 - gamma) * Cyy + gamma * scale_y * eye(size(Cyy));
    Sxx_r = (Sxx_r + Sxx_r') / 2; 
    Syy_r = (Syy_r + Syy_r') / 2; 
    
    Rx = chol(Sxx_r, 'upper');
    Ry = chol(Syy_r, 'upper');
    K = Rx' \ (Cxy / Ry);            
    [Ux, S, Uy] = svd(K, 'econ');
    Vx = Rx \ Ux; 
    Vy = Ry \ Uy; 
end

function C_r = regularize(C)    
    C_r = C + 10e-7 * eye(size(C)) * trace(C) / size(C, 1);
end

function opt = propertylist2struct(varargin)
    opt = [];
    if nargin == 0, return; end
    if isstruct(varargin{1}) || isempty(varargin{1})
        opt = varargin{1};
        iListOffset = 1;
    else
        iListOffset = 0;
    end
    nFields = (nargin - iListOffset) / 2;
    if nFields ~= round(nFields)
        error('Invalid parameter/value list');
    end
    for ff = 1:nFields
        fld = varargin{iListOffset + 2*ff - 1};
        if ~ischar(fld)
            error('Invalid parameter/value list');
        end
        opt.(fld) = varargin{iListOffset + 2*ff};
    end
end

function [opt, isdefault] = set_defaults(opt, varargin)
    isdefault = [];
    if ~isempty(opt)
        for Fld = fieldnames(opt)'
            isdefault.(Fld{1}) = 0;
        end
    end
    defopt = propertylist2struct(varargin{:});
    for Fld = fieldnames(defopt)'
        fld = Fld{1};
        if ~isfield(opt, fld)
            [opt.(fld)] = deal(defopt.(fld));
            isdefault.(fld) = 1;
        end
    end
end