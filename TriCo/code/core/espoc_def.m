function [W, A, corrs] = espoc_def(X_epochs, Z, varargin)
% Extended Source Power Co-modulation (eSPoC) - Recursive Deflation Mode (No Whitening)
opt = propertylist2struct(varargin{:});
opt = set_defaults(opt, ...
                  'X_min_var_explained', 1, ...
                  'cca_mode', 'regularized', ...
                  'cca_reg', 10e-5, ...
                  'n_components', 38, ...
                  'Epochs_cov', []); 
Z = (Z - mean(Z,2)) ./ std(Z,[],2);

% --- 1. Определение рабочих данных (исходные или редуцированные в рекурсии) ---
is_top_level = ~isempty(X_epochs); % Флаг для финальной очистки размерностей
if is_top_level
    [~, n_channels, N_e] = size(X_epochs);
elseif ~isempty(opt.Epochs_cov)
    [n_channels, ~, N_e] = size(opt.Epochs_cov);
else
    error('Either X_epochs or opt.Epochs_cov must be non-empty!');
end

if isempty(opt.Epochs_cov)
    Epochs_cov = zeros(n_channels, n_channels, N_e);
    for ep_idx = 1:N_e
        Epochs_cov(:,:,ep_idx) = cov(X_epochs(:,:,ep_idx));
    end
else
    Epochs_cov = opt.Epochs_cov;
end
Cxx = mean(Epochs_cov, 3);
n_features = (n_channels^2 - n_channels)/2 + n_channels;
Feat = zeros(n_features, N_e);
for ep_idx = 1:N_e
    C = Epochs_cov(:,:,ep_idx);
    Feat(:, ep_idx) = cov2upper(C);
end
Feat = Feat - mean(Feat, 2);

[Featdr, Uf] = project_to_pc(Feat, opt.X_min_var_explained);
if strcmp(opt.cca_mode, 'regularized')
    [Vfdr, Vz_all] = cca(Featdr', Z', opt);
elseif strcmp(opt.cca_mode, 'standard') 
    [Vfdr, Vz_all] = canoncorr(Featdr', Z');
end

v_f = Vfdr(:, 1);
v_z = Vz_all(:, 1);

% ИСПРАВЛЕНИЕ 1: Разделяем математику и сохранение
Vfw = Uf * v_f; % Вектор (массив double) для вычислений

% Передаем в функцию чистый вектор, а не ячейку
[w_best, ~, ~] = project_to_manifold(Vfw, Cxx);
W = w_best; 
Zpr = v_z' * Z;
Env = zeros(1, N_e);
for ep_idx = 1:N_e
    Env(ep_idx) = w_best' * Epochs_cov(:,:,ep_idx) * w_best;
end
corrs = corr(Env', Zpr');

% =========================================================================
% 5. РЕКУРСИВНАЯ ДЕФЛЯЦИЯ
% =========================================================================
if opt.n_components > 1
    opt.n_components = opt.n_components - 1;
    
    B = null(W');
    
    new_dim = size(B, 2);
    Epochs_cov_redux = zeros(new_dim, new_dim, N_e);
    
    for k = 1:N_e
        Epochs_cov_redux(:,:,k) = B' * Epochs_cov(:,:,k) * B;
    end
    
    opt.Epochs_cov = Epochs_cov_redux;
    
    [W_tmp, ~, corrs_tmp] = espoc_def([], Z, opt);
    
    W = [W, B * W_tmp];
    
    corrs = [corrs; corrs_tmp];
end
% =========================================================================
% 6. НОРМАЛИЗАЦИЯ И ВЫЧИСЛЕНИЕ ПАТТЕРНОВ
% =========================================================================
for k = 1:size(W,2)
    W(:,k) = W(:,k) / sqrt(squeeze(W(:,k)' * Cxx * W(:,k)));
end
A = Cxx * W;
if is_top_level && size(W,2) == 1
    corrs = squeeze(corrs);
    W = squeeze(W);
    A = squeeze(A);
end
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [v] = cov2upper(C)
upper_triu_mask = triu(true(size(C)),1);
upper_mask = triu(true(size(C)));
C(upper_triu_mask) = C(upper_triu_mask)*sqrt(2);
upper_triangle = C(upper_mask);
v = upper_triangle(:);
end

function C = upper2cov(v)
n = (-1 + sqrt(1 + 8 * numel(v))) / 2;
C = zeros(n);
upper_mask = triu(true(n));
C(upper_mask) = v;
upper_triu_mask = triu(true(n), 1);
C(upper_triu_mask) = C(upper_triu_mask) / sqrt(2);
C = C + triu(C, 1)';
end

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

function [W, A, s_best] = project_to_manifold(V, Cxx)
    WW = upper2cov(V);
    W_proj = (WW + WW') / 2;
    
    [Uw, S] = eig(W_proj);
    
    [s, idxs] = sort(abs(diag(S)),'descend');
    best_idx = idxs(1);
    
    u_best = Uw(:, best_idx);
    s_best = S(best_idx, best_idx);
    
    wi = u_best;
    
    var_wi = wi' * Cxx * wi;
    
    W = wi / sqrt(var_wi);
    A = Cxx * W; 
end

function opt = propertylist2struct(varargin)
opt= [];
if nargin==0
  return;
end
if isstruct(varargin{1}) || isempty(varargin{1})
  opt= varargin{1};
  iListOffset= 1;
else
  iListOffset = 0;
end
nFields= (nargin-iListOffset)/2;
for ff= 1:nFields
  fld = varargin{iListOffset+2*ff-1};
  opt.(fld)= varargin{iListOffset+2*ff};
end
end

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