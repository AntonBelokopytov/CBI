function [W, A, corrs] = espoc_def(X_epochs, Z, n_components, varargin)
% Extended Source Power Co-modulation with Null-Space Deflation (eSPoC_def)
%
% n_components - сколько компонент нужно извлечь последовательно
opt = propertylist2struct(varargin{:});
opt = set_defaults(opt, ...
                  'X_min_var_explained', 1, ...
                  'whitening_reg', 10e-5, ...
                  'cca_mode', 'regularized', ...
                  'cca_reg', 10e-5);
              
Z = (Z - mean(Z,2)) ./ std(Z,[],2);
[~, n_channels, n_epochs] = size(X_epochs);

Epochs_cov_orig = zeros(n_channels, n_channels, n_epochs);
for ep_idx = 1:n_epochs
    Epochs_cov_orig(:,:,ep_idx) = cov(X_epochs(:,:,ep_idx));
end

W = zeros(n_channels, n_components);
A = zeros(n_channels, n_components);
corrs = zeros(1, n_components);

Epochs_cov_redux = Epochs_cov_orig;
Cxx_full = mean(Epochs_cov_orig, 3);
B_global = eye(n_channels);  

for comp_idx = 1:n_components
    Cxx_redux = mean(Epochs_cov_redux, 3);
    d_current = size(Cxx_redux, 1);
    
    n_features = (d_current^2 - d_current)/2 + d_current;
    Feat = zeros(n_features, n_epochs);
    for ep_idx = 1:n_epochs
        Feat(:, ep_idx) = cov2upper(Epochs_cov_redux(:,:,ep_idx));
    end
    Feat = Feat - mean(Feat, 2);
    
    [Featdr, Uf] = project_to_pc(Feat, opt.X_min_var_explained);
    
    if strcmp(opt.cca_mode, 'regularized')
        [Vfdr, Vz] = cca(Featdr', Z', opt);
    elseif strcmp(opt.cca_mode, 'standard') 
        [Vfdr, Vz] = canoncorr(Featdr', Z');
    end
    
    v_dr = Vfdr(:, 1);
    z_weights = Vz(:, 1);
    v_full = Uf * v_dr;
    
    [w_redux, a_redux] = get_rank1_component(v_full, Cxx_redux);
    
    w_full = B_global * w_redux;
    
    w_full = w_full / sqrt(w_full' * Cxx_full * w_full);
    a_full = Cxx_full * w_full;
    
    W(:, comp_idx) = w_full;
    A(:, comp_idx) = a_full;
    
    Zpr = z_weights' * Z;
    
    p_series = zeros(1, n_epochs);
    for ep_idx = 1:n_epochs
        p_series(ep_idx) = w_redux' * Epochs_cov_redux(:,:,ep_idx) * w_redux;
    end
    corrs(comp_idx) = corr(p_series', Zpr');
    
    if comp_idx < n_components
        B_step = null(a_redux'); 
        
        B_global = B_global * B_step;
        
        d_next = size(B_step, 2);
        Epochs_cov_next = zeros(d_next, d_next, n_epochs);
        for ep_idx = 1:n_epochs
            Epochs_cov_next(:,:,ep_idx) = B_step' * Epochs_cov_redux(:,:,ep_idx) * B_step;
        end
        
        Epochs_cov_redux = Epochs_cov_next;
    end
end
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [w, a] = get_rank1_component(v_full, Cxx)    
    W_proj = upper2cov(v_full);
    W_proj = (W_proj + W_proj') / 2;
    
    [W_eig, D] = eig(W_proj);
    
    [~, idx] = max(abs(diag(D)));
    w = W_eig(:, idx);
    
    scale = sqrt(w' * Cxx * w);
    w = w / scale;
    a = Cxx * w;
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [v] = cov2upper(C)
    upper_triu_mask = triu(true(size(C)),1);
    upper_mask = triu(true(size(C)));
    C(upper_triu_mask) = C(upper_triu_mask)*sqrt(2);
    upper_triangle = C(upper_mask);
    v = upper_triangle(:);
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function C = upper2cov(v)
    n = (-1 + sqrt(1 + 8 * numel(v))) / 2;
    C = zeros(n);
    upper_mask = triu(true(n));
    C(upper_mask) = v;
    upper_triu_mask = triu(true(n), 1);
    C(upper_triu_mask) = C(upper_triu_mask) / sqrt(2);
    C = C + triu(C, 1)';
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [X_proj, U] = project_to_pc(X, min_var_explained)
    X = X - mean(X,2);
    [U,S,~] = svd(X,"econ");
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
    U = U(:,1:n_components);
    X_proj = U' * X;
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [Vx, Vy, Cxx, Cyy] = cca(X, Y, opt)
    gamma = opt.cca_reg;
    X = X - mean(X,1);  
    Y = Y - mean(Y,1);
    [n,~] = size(X);
    Cxx = (X' * X) / (n-1);
    Cyy = (Y' * Y) / (n-1);
    Cxy = (X' * Y) / (n-1);
    scale_x = trace(Cxx) / size(Cxx,1);
    scale_y = trace(Cyy) / size(Cyy,1);
    Sxx_r = (1-gamma)*Cxx + gamma*scale_x*eye(size(Cxx));
    Syy_r = (1-gamma)*Cyy + gamma*scale_y*eye(size(Cyy));
    Sxx_r = (Sxx_r + Sxx_r') / 2; 
    Syy_r = (Syy_r + Syy_r') / 2; 
    Rx = chol(Sxx_r,'upper');
    Ry = chol(Syy_r,'upper');
    K = Rx' \ (Cxy / Ry);            
    [Ux,~,Uy] = svd(K,'econ');
    Vx = Rx \ Ux; 
    Vy = Ry \ Uy; 
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function opt = propertylist2struct(varargin)
    opt= [];
    if nargin==0, return; end
    if isstruct(varargin{1}) | isempty(varargin{1}),
      opt= varargin{1};
      iListOffset= 1;
    else
      iListOffset = 0;
    end
    nFields= (nargin-iListOffset)/2;
    if nFields~=round(nFields), error('Invalid parameter/value list'); end
    for ff= 1:nFields,
      fld = varargin{iListOffset+2*ff-1};
      opt.(fld)= varargin{iListOffset+2*ff};
    end
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [opt, isdefault]= set_defaults(opt, varargin)
    isdefault= [];
    if ~isempty(opt),
      for Fld=fieldnames(opt)', isdefault.(Fld{1})= 0; end
    end
    defopt = propertylist2struct(varargin{:});
    for Fld= fieldnames(defopt)'
      fld= Fld{1};
      if ~isfield(opt, fld)
        [opt.(fld)]= deal(defopt.(fld));
        isdefault.(fld)= 1;
      end
    end
end