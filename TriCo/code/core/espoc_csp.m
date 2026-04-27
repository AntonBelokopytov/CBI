function [W, A, corrs_in, corrs_ex, corrs_in_ex, Zpr_in, Zpr_ex] = espoc_csp(X_epochs, Z, varargin)
% ESPOC_GRAD - Hybrid Euclidean-Riemannian eSPoC for spatial filtering.
% 
% This algorithm extracts spatial filters that maximize the covariation 
% between the power of the filtered EEG/MEG signal and a continuous target variable.
% It uses a hybrid approach:
%   1. Euclidean CCA on vectorized covariances to denoise the target variable.
%   2. Riemannian Tangent Space CSP (Weighted Karcher Means) to robustly 
%      extract the physiological spatial patterns, avoiding the swelling effect.
%
% Usage:
%   [W, A, corrs_in, corrs_ex, corrs_in_ex, Zpr_in, Zpr_ex] = espoc_grad(X_epochs, Z, 'Param1', Value1, ...)
%
% Inputs:
%   X_epochs - [n_channels x n_samples x n_epochs] EEG/MEG data
%   Z        - [1 x n_epochs] or [n_targets x n_epochs] target variable(s)
%
% Options (varargin):
%   'X_min_var_explained' - Variance to keep in PCA (default: 1.0 for all)
%   'cca_mode'            - 'regularized' or 'standard' (default: 'regularized')
%   'cca_reg'             - Regularization parameter for CCA (default: 10e-5)
%
% Outputs:
%   W        - Spatial filters [n_global_src x n_channels x n_local_src]
%   A        - Spatial patterns (forward models)
%   corrs_in - Correlation with the internal (denoised) target variable
%   corrs_ex - Correlation with the external (raw) target variable

    opt = propertylist2struct(varargin{:});
    opt = set_defaults(opt, ...
                      'X_min_var_explained', 1, ...
                      'whitening_reg', 10e-5, ...
                      'cca_mode', 'regularized', ...
                      'cca_reg', 10e-5);
                  
    % Standardize target variable
    Z = (Z - mean(Z, 2)) ./ std(Z, [], 2);
    
    % Step 1: Feature extraction (Euclidean space)
    [Feat, Cxx, Epochs_cov] = get_covariance_series(X_epochs);
    [Featdr, Uf] = project_to_pc(Feat, opt.X_min_var_explained);
    
    % Step 2: Denoising target variable using CCA
    if strcmp(opt.cca_mode, 'regularized')
        [Vfdr, Vz, corrs_in_ex] = cca(Featdr', Z', opt);
    elseif strcmp(opt.cca_mode, 'standard') 
        [Vfdr, Vz] = canoncorr(Featdr', Z');
    end
    
    Vf = Uf * Vfdr;
    n_global_src = size(Vf, 2);
    
    % Step 3: Riemannian spatial filter synthesis
    for global_src_idx = 1:n_global_src        
        z_in_current = Vf(:, global_src_idx)' * Feat;
        
        Zpr_in(global_src_idx, :) = z_in_current;
        Zpr_ex(global_src_idx, :) = Vz(:, global_src_idx)' * Z;
        
        % Riemannian projection to extract clean patterns
        [w, a, s] = project_to_tangent_space(z_in_current, Epochs_cov, Cxx);
        
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
% TANGENT SPACE OPTIMIZATION (RIEMANNIAN CSP / eSPoC)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [W, A, s] = project_to_tangent_space(Z_in, Epochs_cov, Cxx)    
    [n_chan, ~, n_epochs] = size(Epochs_cov);
    
    for i = 1:n_epochs
        Epochs_cov(:,:,i) = regularize(Epochs_cov(:,:,i));
    end
    
    % 1. Split target into positive (activation) and negative (deactivation) poles
    Z_in = Z_in - mean(Z_in);
    Z_pos = max(0, Z_in);  
    Z_neg = max(0, -Z_in); 
    
    sum_pos = max(sum(Z_pos), eps); % Protected against division by zero
    sum_neg = max(sum(Z_neg), eps);
    
    % 2. Baseline extraction (Reference point on the manifold)
    perc_baseline = 0.2; 
    n_base = max(1, round(n_epochs * perc_baseline));
    [~, sort_idx] = sort(abs(Z_in), 'ascend');
    base_idxs = sort_idx(1:n_base);
    
    C_ref = mean(Epochs_cov(:, :, base_idxs), 3);
    C_ref = (C_ref + C_ref') / 2;
    
    % Matrix roots for whitening
    [V_ref, D_ref] = eig(C_ref);
    d_cxx = diag(D_ref);
    d_cxx(d_cxx < eps) = eps; % Protect against negative roots from float precision
    
    C_ref_inv_half = V_ref * diag(1 ./ sqrt(d_cxx)) * V_ref';
    C_ref_half     = V_ref * diag(sqrt(d_cxx)) * V_ref'; 
    
    % Force symmetry
    C_ref_inv_half = real((C_ref_inv_half + C_ref_inv_half') / 2);
    C_ref_half     = real((C_ref_half + C_ref_half') / 2);
    
    A_pos = zeros(n_chan, n_chan);
    A_neg = zeros(n_chan, n_chan);
    
    % 3. Project to tangent space and compute weighted Riemannian means
    for i = 1:n_epochs
        C_i = Epochs_cov(:, :, i);
        C_i = (C_i + C_i') / 2; 
        
        % Whitening
        C_rel = C_ref_inv_half * C_i * C_ref_inv_half;
        C_rel = (C_rel + C_rel') / 2; 
        
        % Riemannian logarithm
        [V_rel, D_rel] = eig(C_rel);
        d_rel = diag(D_rel);
        d_rel(d_rel < eps) = eps;
        
        S_i = V_rel * diag(log(d_rel)) * V_rel';
        S_i = real((S_i + S_i') / 2); 
        
        % Accumulate tangent vectors
        A_pos = A_pos + Z_pos(i) * S_i;
        A_neg = A_neg + Z_neg(i) * S_i;
    end
    
    A_pos = (A_pos + A_pos') / 2 / sum_pos;
    A_neg = (A_neg + A_neg') / 2 / sum_neg;
    
    % 4. Map back to manifold via Riemannian exponential
    C_pos = C_ref_half * expm(A_pos) * C_ref_half;
    C_pos = real((C_pos + C_pos') / 2);
    
    C_neg = C_ref_half * expm(A_neg) * C_ref_half;
    C_neg = real((C_neg + C_neg') / 2);
    
    % 5. Generalized Eigenvalue Problem (CSP)
    C_neg_reg = regularize(C_neg); 
    C_neg_reg = real((C_neg_reg + C_neg_reg') / 2);
    
    [Uw, S] = eig(C_pos, C_neg_reg);    
    [s, idxs] = sort(diag(S), 'descend');
    Uw = Uw(:, idxs);
    
    W = zeros(n_chan, n_chan);
    A = zeros(n_chan, n_chan);
    
    for local_src_idx = 1:n_chan
        wi = real(Uw(:, local_src_idx)); 
        
        % Safeguard normalization
        norm_val = wi' * Cxx * wi;
        if norm_val < eps, norm_val = eps; end
        
        Wprn = wi / sqrt(norm_val);
        W(:, local_src_idx) = Wprn;
        A(:, local_src_idx) = Cxx * Wprn; 
    end
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% HELPER FUNCTIONS
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [v] = cov2upper(C)
    upper_triu_mask = triu(true(size(C)), 1);
    upper_mask = triu(true(size(C)));
    % Scale off-diagonal elements by sqrt(2) to preserve Frobenius norm
    C(upper_triu_mask) = C(upper_triu_mask) * sqrt(2);
    upper_triangle = C(upper_mask);
    v = upper_triangle(:);
end

function [F, Cxx, Epochs_cov] = get_covariance_series(X_epochs)
    [~, n_channels, n_epochs] = size(X_epochs);
    n_features = (n_channels^2 - n_channels) / 2 + n_channels;
    Epochs_cov = zeros(n_channels, n_channels, n_epochs);
    
    for ep_idx = 1:n_epochs
        Epochs_cov(:,:,ep_idx) = cov(X_epochs(:,:,ep_idx));
    end
    
    Cxx = mean(Epochs_cov, 3);
    F = zeros(n_features, n_epochs);
    
    for ep_idx = 1:n_epochs
        F(:, ep_idx) = cov2upper(Epochs_cov(:,:,ep_idx));
    end
    F = F - mean(F, 2);
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
    C_r = C + 10e-5 * eye(size(C)) * trace(C) / size(C, 1);
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% OPTIONS PARSERS
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
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