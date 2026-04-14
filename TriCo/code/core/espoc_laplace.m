function [W, A, corrs] = espoc_laplace(X_epochs, Z, varargin)
opt= propertylist2struct(varargin{:});
opt= set_defaults(opt, ...
                  'X_min_var_explained', 1, ...
                  'whitening_reg', 10e-5, ...
                  'cca_mode', 'regularized', ...
                  'cca_reg', 10e-5);
Z = (Z - mean(Z,2)) ./ std(Z,[],2);
[~,n_channels,n_epochs] = size(X_epochs);

% ---
Epochs_cov = zeros(n_channels,n_channels,n_epochs);
for ep_idx = 1:n_epochs
    Xcov = cov(X_epochs(:,:,ep_idx));
    Epochs_cov(:,:,ep_idx) = Xcov;
end

% Mean covariance matrix
Cxx = mean(Epochs_cov,3);
Epochs_covM = Epochs_cov - Cxx;

Cxz = zeros(n_channels);
for ep_idx = 1:n_epochs
    Cxz = Cxz + Epochs_covM(:,:,ep_idx)*Z(ep_idx);
end

corrs = []; eigenvalues = [];
% Идем в Лапласиан за паттернами!
[w, a, s] = project_to_manifold(Cxz, Cxx, opt);
    
% Find correlation of the filters
Env = [];
for local_src_idx=1:size(w,2)
    for ep_idx=1:size(Epochs_cov,3)
        Env(ep_idx) = w(:,local_src_idx)' * Epochs_cov(:,:,ep_idx) * w(:,local_src_idx);
    end
    cr(local_src_idx)=corr(Env',Z');
end

eigenvalues(1,:) = s;
corrs(1,:) = cr;
W(1,:,:) = w;
A(1,:,:) = a;

if size(W,1)==1
    corrs = squeeze(corrs);
    W = squeeze(W);
    A = squeeze(A);
end
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [W, A, s_best] = project_to_manifold(Cxz, Cxx, opt)
    % Проверки
    if ~isfield(opt, 'dist_matrix') || ~isfield(opt, 'radius')
        error('Для расчета Лапласиана необходимы opt.dist_matrix и opt.radius.');
    end
    
    % Параметр силы сглаживания (alpha). По умолчанию 0.1
    if ~isfield(opt, 'laplace_reg')
        opt.laplace_reg = 1; 
    end
    
    % =======================================================
    % 1. СТРОИМ ФУНКЦИОНАЛЬНО-СТРУКТУРНЫЙ ЛАПЛАСИАН L
    % =======================================================
    
    % --- Шаг А: Физический граф (как было) ---
    P_dist = opt.dist_matrix.^2; 
    
    adj_mask = (opt.dist_matrix < opt.radius);
    adj_mask(1:size(adj_mask,1)+1:end) = 0; % Зануляем диагональ
    
    min_neighbors = 3;
    for i = 1:size(adj_mask, 1)
        if sum(adj_mask(i, :)) < min_neighbors
            [~, sort_idx] = sort(opt.dist_matrix(i, :), 'ascend');
            neighbors = sort_idx(2:min_neighbors+1);
            adj_mask(i, neighbors) = 1;
            adj_mask(neighbors, i) = 1; 
        end
    end
    
    if isfield(opt, 'heat_t')
        t_param = opt.heat_t;
    else
        t_param = mean(P_dist(adj_mask > 0)); 
    end
    
    % Физические веса (чем ближе, тем ближе к 1)
    W_phys = exp(-P_dist / t_param) .* adj_mask;    
    
    % --- Шаг Б: Функциональный граф из Cxz ---
    % Берем модуль Cxz, так как нас интересует СИЛА связи (даже отрицательная ковариация — это связь)
    W_func = abs(Cxz);
    
    % Нормализуем функциональные веса от 0 до 1 (чтобы не сломать масштаб Лапласиана)
    W_func = W_func / max(W_func(:));
    
    % --- Шаг В: Объединение графов ---
    % Умножаем физику на функцию (Hadamard product)
    W_graph = W_phys .* W_func;    
    
    % Создаем итоговый Лапласиан
    D = diag(sum(W_graph, 2));
    L = D - W_graph;

    % =======================================================
    % 2. ПОДГОТОВКА МАТРИЦ ДЛЯ GEVD
    % =======================================================
    % Регуляризация Cxx (защита от вырожденности матриц)
    reg_val = opt.whitening_reg * trace(Cxx) / size(Cxx, 1);
    Cxx_reg = Cxx + reg_val * eye(size(Cxx));
    
    Cxx_inv = inv(Cxx_reg);
    
    % Матрица числителя: проекция Cxz в пространство паттернов
    M_xz = Cxx_inv * Cxz * Cxx_inv;

    % Матрица знаменателя: дисперсия + штраф Лапласиана
    M_xx_L = Cxx_inv + opt.laplace_reg * L;
    
    % =======================================================
    % 3. ОПТИМИЗАЦИЯ
    % =======================================================
    % Ищем паттерны a, которые максимизируют M_xz и минимизируют M_xx_L
    [Ua, S] = eig(M_xz, M_xx_L);
    
    % Сортируем собственные значения по УБЫВАНИЮ (нам нужен максимум cov)
    [s_sorted, idxs] = sort(diag(S), 'descend');
    
    a_best = real(Ua(:, idxs));
    s_best = s_sorted'; 
    
    % =======================================================
    % 4. ВОЗВРАТ К ФИЛЬТРАМ И НОРМАЛИЗАЦИЯ
    % =======================================================
    W = zeros(size(a_best));
    A = zeros(size(a_best));
    
    for c = 1:size(a_best, 2)
        % Извлекаем гладкий паттерн
        a_vec = a_best(:, c);
        
        % Пересчитываем в пространственный фильтр (w = Cxx^-1 * a)
        w_vec = Cxx_inv * a_vec;
        
        % Нормализуем так, чтобы дисперсия отфильтрованного сигнала равнялась 1
        var_w = w_vec' * Cxx_reg * w_vec;
        
        W(:, c) = w_vec / sqrt(var_w);
        A(:, c) = a_vec / sqrt(var_w); 
    end
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

