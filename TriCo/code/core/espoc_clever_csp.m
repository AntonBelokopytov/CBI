function [W, A, corrs_in, corrs_ex, corrs_in_ex, Zpr_in, Zpr_ex] = espoc_clever_csp(X_epochs, Z, varargin)
    opt= propertylist2struct(varargin{:});
    opt= set_defaults(opt, ...
                      'X_min_var_explained', 1, ...
                      'whitening_reg', 10e-5, ...
                      'cca_mode', 'regularized', ...
                      'cca_reg', 10e-5);
                  
    Z = (Z - mean(Z,2)) ./ std(Z,[],2);

    [Feat, Cxx, Epochs_cov] = get_covariance_series(X_epochs);
    [Featdr, Uf] = project_to_pc(Feat, opt.X_min_var_explained);
    
    if strcmp(opt.cca_mode, 'regularized')
        [Vfdr, Vz, corrs_in_ex] = cca(Featdr', Z', opt);
    elseif strcmp(opt.cca_mode, 'standard') 
        [Vfdr, Vz] = canoncorr(Featdr', Z');
    end
    
    Vf = Uf * Vfdr;
    n_global_src = size(Vf,2);
    
    for global_src_idx = 1:n_global_src        
        z_in_current = Vf(:,global_src_idx)' * Feat;
        
        Zpr_in(global_src_idx,:) = z_in_current;
        Zpr_ex(global_src_idx,:) = Vz(:,global_src_idx)' * Z;
        
        [w, a, s] = project_to_tangent_space(z_in_current, Epochs_cov, Cxx, opt);
        
        Env = zeros(1, size(Epochs_cov,3));
        for local_src_idx = 1:size(w,2)
            for ep_idx = 1:size(Epochs_cov,3)
                Env(ep_idx) = w(:,local_src_idx)' * Epochs_cov(:,:,ep_idx) * w(:,local_src_idx);
            end
            cr_in(local_src_idx) = corr(Env', Zpr_in(global_src_idx,:)');
            cr_ex(local_src_idx) = corr(Env', Zpr_ex(global_src_idx,:)');
        end
        
        eigenvalues(global_src_idx,:) = s;
        corrs_in(global_src_idx,:) = cr_in;
        corrs_ex(global_src_idx,:) = cr_ex;
        W(global_src_idx,:,:) = w;
        A(global_src_idx,:,:) = a;
    end
    
    if size(W,1)==1
        corrs_in = squeeze(corrs_in);
        corrs_ex = squeeze(corrs_ex);
        W = squeeze(W);
        A = squeeze(A);
    end
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% ФУНКЦИЯ ОПТИМИЗАЦИИ: TANGENT CCA + HAUFE PATTERN -> MANIFOLD CSP
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [W, A, s] = project_to_tangent_space(Z_in, Epochs_cov, Cxx, opt)    
    % Z_in: целевая переменная (уже очищенная eSPoC), [1 x n_epochs]
    % Epochs_cov: исходные ковариации [n_channels x n_channels x n_epochs]
    
    [n_chan, ~, n_epochs] = size(Epochs_cov);
    for i = 1:n_epochs
        Epochs_cov(:,:,i) = regularize(Epochs_cov(:,:,i));
    end
    
    % 1. Центрируем таргет и разделяем индексы
    Z_in = Z_in - mean(Z_in);

    idx_pos = find(Z_in > 0);
    idx_neg = find(Z_in < 0);
    
    Z_pos = Z_in(idx_pos);
    Z_neg = -Z_in(idx_neg); % берем по модулю для корректной ковариации
    
    % 2. Бейзлайн (референсная точка)
    perc_baseline = 0.2; 
    n_base = max(1, round(n_epochs * perc_baseline));
    [~, sort_idx] = sort(abs(Z_in), 'ascend');
    Cref = mean(Epochs_cov(:,:,sort_idx(1:n_base)), 3);
    
    % Подготовка матриц
    [V_cxx, D_cxx] = eig(regularize(Cref));
    d_cxx = diag(D_cxx);
    Cxx_inv_half = V_cxx * diag(1 ./ sqrt(d_cxx)) * V_cxx';
    Cxx_half     = V_cxx * diag(sqrt(d_cxx)) * V_cxx';
    
    % 3. Проецируем в касательное пространство и векторизуем
    n_features = (n_chan^2 - n_chan)/2 + n_chan;
    S_vec = zeros(n_features, n_epochs);
    
    for i = 1:n_epochs
        C_i = (Epochs_cov(:,:,i) + Epochs_cov(:,:,i)') / 2; 
        C_rel = Cxx_inv_half * C_i * Cxx_inv_half;
        C_rel = (C_rel + C_rel') / 2; 
        
        [V_rel, D_rel] = eig(regularize(C_rel));
        d_rel = diag(D_rel); d_rel(d_rel < eps) = eps;
        
        % Риманов логарифм (матрица в касательном пространстве)
        S_i = V_rel * diag(log(d_rel)) * V_rel';
        
        % Векторизация (чтобы подать в CCA и cov)
        S_vec(:, i) = cov2upper(S_i);
    end
    
    % 4. РАЗДЕЛЕНИЕ НА ПОЛЮСА
    S_pos = S_vec(:, idx_pos);
    S_neg = S_vec(:, idx_neg);
    
    % 5. TANGENT CCA (Поиск направлений корреляции внутри касательного пространства)
    % Используем регуляризованный CCA для предотвращения переобучения на малых выборках
    [V_pos, ~] = cca(S_pos', Z_pos', opt); 
    [V_neg, ~] = cca(S_neg', Z_neg', opt);
    
    % 6. ПРАВИЛО ХАУФЕ В КАСАТЕЛЬНОМ ПРОСТРАНСТВЕ (Ваша идея!)
    % A = cov(X) * W
    A_pos_vec = cov(S_pos') * V_pos(:, 1);
    A_neg_vec = cov(S_neg') * V_neg(:, 1);
    
    % Девекторизуем обратно в квадратные симметричные матрицы
    A_pos = upper2cov(A_pos_vec);
    A_neg = upper2cov(A_neg_vec);
    
    A_pos = (A_pos + A_pos') / 2;
    A_neg = (A_neg + A_neg') / 2;
    
    % 7. Возврат на физическое многообразие (expm)
    C_pos = Cxx_half * expm(A_pos) * Cxx_half;
    C_pos = (C_pos + C_pos') / 2;
    
    C_neg = Cxx_half * expm(A_neg) * Cxx_half;
    C_neg = (C_neg + C_neg') / 2;
    
    % 8. Классический CSP: Контраст паттерна активации и паттерна дезактивации
    [Uw, S] = eig(C_pos, regularize(C_neg));    
    [s, idxs] = sort(diag(S), 'descend');
    Uw = Uw(:, idxs);
    
    W = zeros(n_chan, n_chan);
    A = zeros(n_chan, n_chan);
    
    for local_src_idx = 1:n_chan
        wi = Uw(:, local_src_idx); 
        Wprn = wi / sqrt(wi' * Cxx * wi); 
        
        W(:, local_src_idx) = Wprn;
        A(:, local_src_idx) = Cxx * Wprn; 
    end
end

function [v] = cov2upper(C)

upper_triu_mask = triu(true(size(C)),1);
upper_mask = triu(true(size(C)));
C(upper_triu_mask) = C(upper_triu_mask)*sqrt(2);
upper_triangle = C(upper_mask);
v = upper_triangle(:);

end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function [F, Cxx, Epochs_cov] = get_covariance_series(X_epochs)

% Function to get upper triangular covarience time series in dimension
% reduced space

[~,n_channels,n_epochs] = size(X_epochs);
n_features = (n_channels^2-n_channels)/2+n_channels;

Epochs_cov = zeros(n_channels,n_channels,n_epochs);
for ep_idx = 1:n_epochs
    Xcov = cov(X_epochs(:,:,ep_idx));
    Epochs_cov(:,:,ep_idx) = Xcov;
end
% Mean covariance matrix
Cxx = mean(Epochs_cov,3);

F = zeros(n_features,n_epochs);
for ep_idx = 1:n_epochs
    Xcov = Epochs_cov(:,:,ep_idx);
    F(:, ep_idx) = cov2upper(Xcov);
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

% Project
X_proj = U' * X;

end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function [Vx, Vy, S] = cca(X, Y, opt)

% Regularized CCA

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
[Ux,S,Uy] = svd(K,'econ');

Vx = Rx \ Ux; 
Vy = Ry \ Uy; 

end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function C_r = regularize(C)    
    C_r = C + 10e-5 * eye(size(C)) * trace(C)/size(C,1);
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% OTHER HELPERS
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function opt = propertylist2struct(varargin)
% PROPERTYLIST2STRUCT - Make options structure from parameter/value list
%
%   OPT= propertylist2struct('param1', VALUE1, 'param2', VALUE2, ...)
%   Generate a structure OPT with fields 'param1' set to value VALUE1, field
%   'param2' set to value VALUE2, and so forth.
%
%   See also set_defaults

opt= [];
if nargin==0,
  return;
end

if isstruct(varargin{1}) | isempty(varargin{1}),
  % First input argument is already a structure: Start with that, write
  % the additional fields
  opt= varargin{1};
  iListOffset= 1;
else
  % First argument is not a structure: Assume this is the start of the
  % parameter/value list
  iListOffset = 0;
end

nFields= (nargin-iListOffset)/2;
if nFields~=round(nFields),
  error('Invalid parameter/value list');
end

for ff= 1:nFields,
  fld = varargin{iListOffset+2*ff-1};
  if ~ischar(fld),
    error('Invalid parameter/value list');
  end
%  prp= varargin{iListOffset+2*ff};
%  opt= setfield(opt, fld, prp);
  opt.(fld)= varargin{iListOffset+2*ff};
end

end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function [opt, isdefault]= set_defaults(opt, varargin)
%[opt, isdefault]= set_defaults(opt, field/value list)
%
%Description:
% This functions fills in the given struct opt some new fields with
% default values, but only when these fields DO NOT exist before in opt.
% Existing fields are kept with their original values.
%
%Example:
%   opt= set_defaults(opt, 'color','g', 'linewidth',3);
%
% The second output argument isdefault is a struct with the same fields
% as the returned opt, where each field has a boolean value indicating
% whether or not the default value was inserted in opt for that field.

% blanker@cs.tu-berlin.de

% Set 'isdefault' to ones for the field already present in 'opt'
isdefault= [];
if ~isempty(opt),
  for Fld=fieldnames(opt)',
    isdefault.(Fld{1})= 0;
  end
end

defopt = propertylist2struct(varargin{:});
for Fld= fieldnames(defopt)',
  fld= Fld{1};
  if ~isfield(opt, fld),
    %% if opt is a struct *array*, the fields of all elements need to
    %% be set. This is done with the 'deal' function.
    [opt.(fld)]= deal(defopt.(fld));
    isdefault.(fld)= 1;
  end
end

end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
