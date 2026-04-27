function [W, A, corrs_in, corrs_ex, corrs_in_ex, Zpr_in, Zpr_ex] = espoc_riemann(X_epochs, Z, varargin)
    opt= propertylist2struct(varargin{:});
    opt= set_defaults(opt, ...
                      'X_min_var_explained', 1, ...
                      'whitening_reg', 10e-5, ...
                      'cca_mode', 'regularized', ...
                      'cca_reg', 10e-5);
                  
    Z = (Z - mean(Z,2)) ./ std(Z,[],2);
    [~, n_chan, n_epochs] = size(X_epochs);
    
    % 1. Вычисление исходных ковариационных матриц
    Epochs_cov = zeros(n_chan, n_chan, n_epochs);
    for ep_idx = 1:n_epochs
        Epochs_cov(:,:,ep_idx) = regularize(cov(X_epochs(:,:,ep_idx)));
    end
    
    % 2. Выбор референсной точки (глобальное среднее)
    % Можно заменить на riemann_mean или среднее по фону (baseline)
    Cref = mean(Epochs_cov, 3);
    
    [V_cxx, D_cxx] = eig(regularize(Cref));
    d_cxx = diag(D_cxx);
    Cxx_inv_half = V_cxx * diag(1 ./ sqrt(d_cxx)) * V_cxx';
    Cxx_half     = V_cxx * diag(sqrt(d_cxx)) * V_cxx';
    
    % 3. Проекция в касательное пространство и векторизация
    n_features = (n_chan^2 - n_chan)/2 + n_chan;
    Feat_tgt = zeros(n_features, n_epochs);
    S_matrices = zeros(n_chan, n_chan, n_epochs); % Храним матрицы S_i для шага 6
    
    for i = 1:n_epochs
        C_i = Epochs_cov(:,:,i);
        C_i = (C_i + C_i') / 2;
        
        C_rel = Cxx_inv_half * C_i * Cxx_inv_half;
        C_rel = (C_rel + C_rel') / 2;
        
        [V_rel, D_rel] = eig(regularize(C_rel));
        d_rel = diag(D_rel);
        d_rel(d_rel < eps) = eps;
        S_i = V_rel * diag(log(d_rel)) * V_rel';
        S_i = (S_i + S_i') / 2;
        
        S_matrices(:,:,i) = S_i;
        
        % Векторизация с сохранением изометрии Фробениуса (sqrt(2) вне диагонали)
        Feat_tgt(:, i) = cov2upper(S_i);
    end
    
    Feat_tgt = Feat_tgt - mean(Feat_tgt, 2);
    
    % 4. Уменьшение размерности (PCA) в касательном пространстве
    [Featdr, Uf] = project_to_pc(Feat_tgt, opt.X_min_var_explained);
    
    % 5. CCA: Поиск внутренней компоненты
    if strcmp(opt.cca_mode, 'regularized')
        [Vfdr, Vz, corrs_in_ex] = cca(Featdr', Z', opt);
    elseif strcmp(opt.cca_mode, 'standard') 
        [Vfdr, Vz] = canoncorr(Featdr', Z');
    end
    
    Vf = Uf * Vfdr;
    n_global_src = size(Vf,2);
    
    % Предварительная аллокация
    Zpr_in = zeros(n_global_src, n_epochs);
    Zpr_ex = zeros(n_global_src, n_epochs);
    
    % 6. Синтез пространственных паттернов для каждой найденной компоненты
    for global_src_idx = 1:n_global_src        
        % Внутренняя (очищенная) переменная
        z_in_current = Vf(:,global_src_idx)' * Feat_tgt;
        
        Zpr_in(global_src_idx,:) = z_in_current;
        Zpr_ex(global_src_idx,:) = Vz(:,global_src_idx)' * Z;
        
        % Разделяем на полюса активации и дезактивации
        z_c = z_in_current - mean(z_in_current);
        z_pos = max(0, z_c);
        z_neg = max(0, -z_c);
        
        sum_pos = sum(z_pos); if sum_pos == 0, sum_pos = 1; end
        sum_neg = sum(z_neg); if sum_neg == 0, sum_neg = 1; end
        
        A_pos = zeros(n_chan, n_chan);
        A_neg = zeros(n_chan, n_chan);
        
        % Взвешиваем касательные векторы (они уже вычислены в шаге 3!)
        for i = 1:n_epochs
            A_pos = A_pos + (z_pos(i) / sum_pos) * S_matrices(:,:,i);
            A_neg = A_neg + (z_neg(i) / sum_neg) * S_matrices(:,:,i);
        end
        
        % Возврат на многообразие через экспоненту
        C_pos = Cxx_half * expm(A_pos) * Cxx_half;
        C_pos = (C_pos + C_pos') / 2;
        C_neg = Cxx_half * expm(A_neg) * Cxx_half;
        C_neg = (C_neg + C_neg') / 2;
        
        % Классический CSP (GEVP)
        [Uw, S_csp] = eig(C_pos, regularize(C_neg));    
        [s, idxs] = sort(diag(S_csp), 'descend');
        Uw = Uw(:, idxs);
        
        w = zeros(n_chan, n_chan);
        a = zeros(n_chan, n_chan);
        
        for local_src_idx = 1:n_chan
            wi = Uw(:, local_src_idx); 
            Wprn = wi / sqrt(wi' * Cref * wi);
            w(:, local_src_idx) = Wprn;
            a(:, local_src_idx) = Cref * Wprn; 
        end
        
        % Оценка огибающих и расчет корреляций
        Env = zeros(1, n_epochs);
        cr_in = zeros(1, n_chan);
        cr_ex = zeros(1, n_chan);
        
        for local_src_idx = 1:size(w,2)
            for ep_idx = 1:n_epochs
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
% HELPER FUNCTIONS
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [v] = cov2upper(C)
    upper_triu_mask = triu(true(size(C)),1);
    upper_mask = triu(true(size(C)));
    % Умножаем недиагональные элементы на sqrt(2) для сохранения нормы Фробениуса!
    C(upper_triu_mask) = C(upper_triu_mask)*sqrt(2);
    upper_triangle = C(upper_mask);
    v = upper_triangle(:);
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

function [Vx, Vy, S] = cca(X, Y, opt)
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

function C_r = regularize(C)    
    C_r = C + 10e-5 * eye(size(C)) * trace(C)/size(C,1);
end

function opt = propertylist2struct(varargin)
    opt= [];
    if nargin==0, return; end
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