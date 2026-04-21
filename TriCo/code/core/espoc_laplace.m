function [W, A, corrs] = espoc_laplace(X_epochs, Z, Dist, varargin)
opt = propertylist2struct(varargin{:});
opt = set_defaults(opt, ...
                  'X_min_var_explained', 1, ...
                  'whitening_reg', 10e-5, ...
                  'cca_mode', 'regularized', ...
                  'cca_reg', 10e-5, ...
                  'laplace_t', 10, ...         % Параметр t (ширина Гауссова ядра)
                  'laplace_radius', 10);        % Радиус (epsilon) для отсечения соседей

Z = (Z - mean(Z,2)) ./ std(Z,[],2);
% ---
[Feat, Wm, Cxx, Epochs_cov, ~] = get_white_covariance_series(X_epochs, opt);
Cff = cov(Feat');
[Featdr, Uf] = project_to_pc(Feat, opt.X_min_var_explained);

if strcmp(opt.cca_mode, 'regularized')
    [Vfdr, Vz] = cca(Featdr', Z', opt);
elseif strcmp(opt.cca_mode, 'standard') 
    [Vfdr, Vz] = canoncorr(Featdr', Z');
end

% Return found filters from dimension reduced space 
Vfw = Uf * Vfdr;
Afw = Cff * Vfw;

% Вычисление Лапласиана по методу Белкина с учетом радиуса
L = compute_laplacian(Dist, opt.laplace_t, opt.laplace_radius);

% Project and normalize EEG/MEG filters
for global_src_idx = 1:size(Afw,2)
    [w, a, s] = project_to_manifold(Afw(:,global_src_idx), L, Cxx, opt);
    
    % Project target variable to its CCA component 
    Zpr = Vz(:,global_src_idx)' * Z;
    
    % Find correlation of the filters
    Env = zeros(1, size(Epochs_cov,3));
    cr = zeros(1, size(w,2));
    
    for local_src_idx = 1:size(w,2)
        for ep_idx = 1:size(Epochs_cov,3)
            Env(ep_idx) = w(:,local_src_idx)' * Epochs_cov(:,:,ep_idx) * w(:,local_src_idx);
        end
        cr(local_src_idx) = corr(Env', Zpr');
    end
    
    % Опционально: можно отсортировать решения внутри глобальной компоненты по корреляции
    % [cr, idx] = sort(abs(cr), 'descend');
    % w = w(:,idx);
    % a = a(:,idx);
    % s = s(idx);
    
    eigenvalues(global_src_idx,:) = s;
    corrs(global_src_idx,:) = cr;
    W(global_src_idx,:,:) = w;
    A(global_src_idx,:,:) = a;
end

if size(W,1) == 1
    corrs = squeeze(corrs);
    W = squeeze(W);
    A = squeeze(A);
end

end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function L = compute_laplacian(Dist, t, radius)
    W = exp(-(Dist.^2) / t);
    W(Dist > radius) = 0;
    W = W - diag(diag(W));
    D_deg = diag(sum(W, 2));
    L = D_deg - W;
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [W, A, s] = project_to_manifold(V, L, Cxx, opt)    
    Cxx_r = Cxx + opt.whitening_reg * eye(size(Cxx)) * trace(Cxx) / size(Cxx,1);
    Cxx_r = (Cxx_r + Cxx_r') / 2; 

    AA = upper2cov(V);
    AA = (AA + AA') / 2;
    
    % Стабилизация проекции
    AA_r = AA + opt.whitening_reg * eye(size(AA)) * trace(AA) / size(AA,1);
    AA_r = (AA_r + AA_r') / 2;
    
    % Ищем пространственные паттерны 'Ap', которые гладкие (по Лапласиану L), 
    % но объясняют максимум дисперсии в матрице ковариации паттерна 'AA_r'
    % L = Cxx_r \ L / Cxx_r;
    [Ap, S] = eig(AA_r, Cxx_r);
    
    % ВАЖНО: Самые гладкие паттерны соответствуют МИНИМАЛЬНЫМ собственным значениям
    [s, idxs] = sort(diag(S), 'descend');
    Ap = Ap(:, idxs);
    
    n_channels = size(Cxx, 1);
    n_local_src = size(Ap, 2);
    W = zeros(n_channels, n_local_src);
    A = zeros(n_channels, n_local_src);
    
    for local_src_idx = 1:n_local_src
        a_i = Ap(:, local_src_idx); 
        
        % Восстанавливаем фильтр из найденного гладкого паттерна
        w_i =  a_i;
        
        % Нормализуем фильтр относительно данных
        w_norm = w_i / sqrt(abs(w_i' * Cxx * w_i));
        
        % Пересчитываем итоговый паттерн прямой модели
        W(:, local_src_idx) = w_norm;
        A(:, local_src_idx) = Cxx_r * w_norm; 
    end
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
function [F, Wm, Cxx, Epochs_cov, Epochs_covW] = get_white_covariance_series(X_epochs, opt)
    [~,n_channels,n_epochs] = size(X_epochs);
    n_features = (n_channels^2-n_channels)/2+n_channels;
    Epochs_cov = zeros(n_channels,n_channels,n_epochs);
    for ep_idx = 1:n_epochs
        Xcov = cov(X_epochs(:,:,ep_idx));
        Epochs_cov(:,:,ep_idx) = Xcov;
    end
    Cxx = mean(Epochs_cov,3);
    
    % В данном скрипте отбеливание не используется напрямую, 
    % так как eSPoC может работать и без него.
    Wm = eye(n_channels);
    
    Epochs_covW = zeros(n_channels,n_channels,n_epochs);
    F = zeros(n_features,n_epochs);
    for ep_idx = 1:n_epochs
        XcovW = Wm * Epochs_cov(:,:,ep_idx) * Wm';
        Epochs_covW(:,:,ep_idx) = XcovW;
        F(:, ep_idx) = cov2upper(XcovW);
    end
    F = F - mean(F,2);
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
    if nargin==0,
      return;
    end
    if isstruct(varargin{1}) || isempty(varargin{1})
      opt= varargin{1};
      iListOffset= 1;
    else
      iListOffset = 0;
    end
    nFields= (nargin-iListOffset)/2;
    if nFields~=round(nFields)
      error('Invalid parameter/value list');
    end
    for ff= 1:nFields
      fld = varargin{iListOffset+2*ff-1};
      if ~ischar(fld)
        error('Invalid parameter/value list');
      end
      opt.(fld)= varargin{iListOffset+2*ff};
    end
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [opt, isdefault]= set_defaults(opt, varargin)
    isdefault= [];
    if ~isempty(opt)
      for Fld=fieldnames(opt)'
        isdefault.(Fld{1})= 0;
      end
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