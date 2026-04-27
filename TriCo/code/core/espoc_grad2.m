function [W, A, corrs_in, corrs_ex, corrs_in_ex, Zpr_in, Zpr_ex] = espoc_grad2(X_epochs, Z, varargin)
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
        
        [w, a, s] = project_to_tangent_space(z_in_current, Epochs_cov, Cxx);
        
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
% ФУНКЦИЯ ОПТИМИЗАЦИИ В КАСАТЕЛЬНОМ ПРОСТРАНСТВЕ (ПОЛЮСНЫЙ Tangent SPoC)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [W, A, s] = project_to_tangent_space(Z_in, Epochs_cov, Cxx)    
    % Z_in: целевая переменная (ожидается центрированной!), [1 x n_epochs]
    % Epochs_cov: исходные ковариации [n_channels x n_channels x n_epochs]
    % Cxx: средняя ковариационная матрица (глобальная референсная точка)
    
    [n_chan, ~, n_epochs] = size(Epochs_cov);
    
    % Регуляризация (защита от вырожденности матриц)
    for i = 1:n_epochs
        Epochs_cov(:,:,i) = regularize(Epochs_cov(:,:,i));
    end
    Z_in = Z_in - mean(Z_in);
    
    % 1. Формируем веса для полюсов активации и дезактивации
    Z_pos = Z_in; Z_pos(Z_pos < 0) = 0;
    Z_neg = -Z_in; Z_neg(Z_neg < 0) = 0;
    
    C_pos = zeros(n_chan, n_chan);
    C_neg = zeros(n_chan, n_chan);
    
    % Взвешиваем векторы/матрицы ДО проекции
    for i = 1:n_epochs
        C_pos = C_pos + Z_pos(i) * Epochs_cov(:,:,i);
        C_neg = C_neg + Z_neg(i) * Epochs_cov(:,:,i);
    end
    
    % Нормируем на сумму весов (получаем взвешенные средние)
    % Примечание: здесь используется Евклидово взвешенное среднее, 
    % но его можно заменить на взвешенный Karcher Mean.
    C_pos = C_pos / sum(Z_pos);
    C_neg = C_neg / sum(Z_neg);
    
    C_pos = (C_pos + C_pos') / 2;
    C_neg = (C_neg + C_neg') / 2;
    
    % 2. Определяем референсную точку (C_ref)
    % Берем эпохи с минимальным значением Z (базлайн)
    perc_baseline = 0.2; 
    n_base = max(1, round(n_epochs * perc_baseline));
    [~, sort_idx] = sort(abs(Z_in), 'ascend');
    base_idxs = sort_idx(1:n_base);
    
    % В качестве референса можно использовать глобальную Cxx 
    % или усредненный базлайн. Оставим базлайн:
    Cref = mean(Epochs_cov(:,:,base_idxs), 3);
    Cref = (Cref + Cref') / 2;
    
    % Находим Cref^{-1/2} для перехода в касательное пространство
    [V_cxx, D_cxx] = eig(regularize(Cref));
    d_cxx = diag(D_cxx);
    d_cxx(d_cxx < eps) = eps;
    Cref_inv_half = V_cxx * diag(1 ./ sqrt(d_cxx)) * V_cxx';
    
    % 3. Проецируем наши "полюса" на касательное многообразие (logm)
    
    % Проекция полюса активации
    C_pos_rel = Cref_inv_half * C_pos * Cref_inv_half;
    C_pos_rel = (C_pos_rel + C_pos_rel') / 2; 
    [V_pos, D_pos] = eig(C_pos_rel);
    d_pos = diag(D_pos); d_pos(d_pos < eps) = eps;
    S_pos = V_pos * diag(log(d_pos)) * V_pos';
    
    % Проекция полюса дезактивации
    C_neg_rel = Cref_inv_half * C_neg * Cref_inv_half;
    C_neg_rel = (C_neg_rel + C_neg_rel') / 2; 
    [V_neg, D_neg] = eig(C_neg_rel);
    d_neg = diag(D_neg); d_neg(d_neg < eps) = eps;
    S_neg = V_neg * diag(log(d_neg)) * V_neg';
    
    % 4. Делаем CSP относительно референсной точки
    % Поскольку мы в отбеленном касательном пространстве, 
    % фоновый шум равен единичной матрице, и CSP - это просто разность:
    A_diff = S_pos - S_neg;
    A_diff = (A_diff + A_diff') / 2;
    
    [Uw, S] = eig(A_diff);    
    [s, idxs] = sort(diag(S), 'descend');
    Uw = Uw(:, idxs);
    
    W = zeros(n_chan, n_chan);
    A = zeros(n_chan, n_chan);
    
    % 5. Возврат фильтров и паттернов в физическое пространство ЭЭГ
    for local_src_idx = 1:n_chan
        wi = Cref_inv_half * Uw(:, local_src_idx); 
        
        Wprn = wi / sqrt(wi' * Cxx * wi);
        
        W(:, local_src_idx) = Wprn;
        A(:, local_src_idx) = Cxx * Wprn; 
    end
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% ФУНКЦИЯ ВЫЧИСЛЕНИЯ РИМАНОВА СРЕДНЕГО (Karcher Mean)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function M = riemann_mean(Covs, tol, max_iter)
    % Covs: 3D массив ковариационных матриц [n_chan x n_chan x n_epochs]
    % tol: порог сходимости (по умолчанию 1e-5)
    % max_iter: макс. число итераций (по умолчанию 50)
    
    if nargin < 2, tol = 1e-5; end
    if nargin < 3, max_iter = 50; end
    
    [n_chan, ~, n_epochs] = size(Covs);
    
    % 1. Инициализация: начинаем с обычного евклидова среднего
    M = mean(Covs, 3); 
    
    for iter = 1:max_iter
        % Принудительная симметризация (защита от ошибок округления)
        M = (M + M') / 2; 
        
        % Разложение текущего среднего M для перехода в касательное пространство
        [V, D] = eig(M);
        d = diag(D);
        d(d < eps) = eps; % Защита от нулей/отрицательных значений
        
        % Нам понадобятся M^(-1/2) и M^(1/2)
        M_inv_half = V * diag(1 ./ sqrt(d)) * V';
        M_half = V * diag(sqrt(d)) * V';
        
        tangent_mean = zeros(n_chan, n_chan);
        
        % 2. Проецируем все матрицы в касательное пространство ВОКРУГ текущего M
        for i = 1:n_epochs
            C_i = Covs(:,:,i);
            C_i = (C_i + C_i') / 2;
            
            % Отбеливание (относительная ковариация)
            C_rel = M_inv_half * C_i * M_inv_half;
            C_rel = (C_rel + C_rel') / 2;
            
            [Vc, Dc] = eig(C_rel);
            dc = diag(Dc);
            dc(dc < eps) = eps;
            
            % Риманов логарифм (это вектор в касательном пространстве)
            log_C_rel = Vc * diag(log(dc)) * Vc';
            
            % Суммируем векторы
            tangent_mean = tangent_mean + log_C_rel;
        end
        
        % 3. Ищем среднее в касательном пространстве (это и есть градиент)
        tangent_mean = tangent_mean / n_epochs;
        
        % Проверка сходимости: если градиент исчез, значит мы в центре масс!
        grad_norm = norm(tangent_mean, 'fro');
        if grad_norm < tol
            % disp(['Риманово среднее найдено за ', num2str(iter), ' итераций.']);
            break;
        end
        
        % 4. Возврат на многообразие (Exponential map)
        % Двигаемся из M в направлении градиента
        [Vt, Dt] = eig(tangent_mean);
        exp_tangent = Vt * diag(exp(diag(Dt))) * Vt';
        
        % Обновляем центральную точку
        M = M_half * exp_tangent * M_half;
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
