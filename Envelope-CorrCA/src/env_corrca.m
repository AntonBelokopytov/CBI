<<<<<<< Updated upstream
function [W, A, z_trials, X_epochs, raw_var] = env_corrca(X, Fs, Wsize, Ssize, lambda)
    if nargin < 5
        lambda = 1e-5; 
    end

    [~, n_channels, n_trials] = size(X);
    
    for tr_idx=1:n_trials
        X(:,:,tr_idx) = X(:,:,tr_idx) - mean(X(:,:,tr_idx),1);
    end
    Xmean = mean(X,3);
    
    temp_epo = epoch_data(X(:,:,1), Fs, Wsize, Ssize);
    [~, ~, n_epochs] = size(temp_epo);
    X_epochs = zeros(size(temp_epo,1), n_channels, n_epochs, n_trials);
    for tr_idx=1:n_trials
        mX = X(:,:,tr_idx) - Xmean;
        mX = mX ./ sqrt(trace(cov(mX)));
        X_epochs(:,:,:,tr_idx) = epoch_data(mX,Fs,Wsize,Ssize);
    end
    
    [~, ~, n_epochs, ~] = size(X_epochs);
    X_covs = zeros(n_channels, n_channels, n_epochs, n_trials);
    for j=1:n_trials
        for i=1:n_epochs
            X_covs(:,:,i,j) = cov(X_epochs(:,:,i,j));
=======
function [W, A] = env_corrca(X, ...
Fs, Wsize, Ssize)

[~, ~, n_trials] = size(X);

for tr_idx=1:n_trials
    X(:,:,tr_idx) = X(:,:,tr_idx) - mean(X(:,:,tr_idx),1);
end
Xmean = mean(X,3);

X_epochs = [];
for tr_idx=1:n_trials
    mX = X(:,:,tr_idx) - Xmean;
    mX = mX ./ sqrt(trace(cov(mX)));
    X_epochs(:,:,:,tr_idx) = epoch_data(mX,Fs,Wsize,Ssize);
end

[~, ~, n_epochs, ~] = size(X_epochs);
X_covs = [];
for i=1:n_epochs
    for j=1:n_trials
        X_covs(:,:,i,j) = cov(X_epochs(:,:,i,j));
    end
end
mX_covs = mean(X_covs,4);
Cm = mean(mX_covs,3);
Wm = Cm^-0.5;

X_covsVecW = [];
for i=1:n_epochs
    for j=1:n_trials
        X_covsVecW(i,:,j) = cov2upper(Wm*X_covs(:,:,i,j)*Wm');
    end
end

[Vc, ~, ~] = corrca(X_covsVecW);
z = mean(X_covsVecW,3) * Vc;
z = (z - mean(z,1)) ./ std(z,[],1);

Af = mean(X_covsVecW,3)' * z;

W = []; A = [];
for i=1:size(Vc,2)
    [w, a, s] = project_filters_to_manifold(Af(:,i), Wm, Cm);
    W(i,:,:) = w;
    A(i,:,:) = a;
    eigenvals(i,:) = s;
end

% corrs = intertr_corrs(W, X_covs, n_plot_comp);
% visualize(z, eigenvals, corrs, n_plot_comp, Wsize, Ssize)

end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function X_epo = epoch_data(X, Fs, Ws, Ss)

W = fix(Ws*Fs);
S = fix(Ss*Fs);
range = 1:W; ep = 1;
X_epo = [];
while range(end) <= size(X,1)
    X_epo(:,:,ep) = X(range,:); 
    range = range + S; ep = ep + 1;
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
    assert(mod(n,1) == 0, 'Vector length does not correspond to a triangular matrix.');

    C = zeros(n);
    upper_mask = triu(true(n));
    C(upper_mask) = v;

    upper_triu_mask = triu(true(n), 1);
    C(upper_triu_mask) = C(upper_triu_mask) / sqrt(2);

    C = C + triu(C, 1)';
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function [W, Rw, Rb] = corrca(X)

gamma = 0.; % shrinkage parameter

[T, D, N] = size(X);

Xc = X - mean(X,1);

% инициализация
Rw = zeros(D,D);
Rb = zeros(D,D);

for i = 1:N
    
    Xi = squeeze(Xc(:,:,i)); % T × D
    
    % within-subject covariance
    Ci = (Xi' * Xi) / (T-1);
    Rw = Rw + Ci;
    
    % between-subject covariance
    for j = i+1:N
        
        Xj = squeeze(Xc(:,:,j));
        
        Cij = (Xi' * Xj) / (T-1);
        Rb = Rb + Cij;
    end
end

% нормировка
Rw = Rw / N;
Rb = (Rb + Rb') / (N*(N-1));

% shrinkage regularization
Rw_reg = (1-gamma)*Rw + gamma*mean(eig(Rw))*eye(D);

% generalized eigenvalue problem
[W, S] = eig(Rb, Rw_reg, 'chol');

% сортировка
[S, indx] = sort(diag(S), 'descend');
W = W(:, indx);

end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function [W, A, S] = project_filters_to_manifold(V, Wm, Cxx)

% Project filters to manifold
WW = upper2cov(V);

[Uw,S] = eig(WW);S=diag(S);[S,idxs]=sort(S,'descend');Uw=Uw(:,idxs);
% [Uw,S,~] = svd(WW);s=diag(S);
% stem(s)
% xlabel('number of component')
% ylabel('\lambda value')
% title('Spectrum of eigenvalues of the matrix W')
% Optionally svd() could be used instead of eig() (Result is the same. Order differs)
% [Uw,~,~] = svd(WW);

% Normalization and pattern recovery
for local_src_idx=1:size(Uw,2)
    % Return filters from the whightened space
    wi = Wm * Uw(:,local_src_idx);
    % Normalize
    Wprn = wi / sqrt(wi' * Cxx * wi);
    W(:,local_src_idx) = Wprn;
    A(:,local_src_idx) = Cxx * Wprn / (Wprn' * Cxx * Wprn);
end

end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function corrs = intertr_corrs(W, X_covs, n_filters_to_eval)    
    % W: [n_filters, n_channels, n_components]
    % X_covs: [n_channels, n_channels, n_epochs, n_trials]
    % n_filters_to_eval: сколько первых наборов фильтров оценивать
    
    [n_filters, ~, n_components] = size(W);
    [~, ~, n_epochs, n_trials] = size(X_covs);
    
    % Выбираем минимум, чтобы не выйти за границы массива
    n_to_do = min(n_filters_to_eval, n_filters);
    
    % Результирующая матрица: строки — выбранные фильтры, столбцы — все их компоненты
    corrs = zeros(n_to_do, n_components);

    for f_idx = 1:n_to_do
        for comp_idx = 1:n_components
            Envs = zeros(n_epochs, n_trials);
            
            % Извлекаем веса
            w = squeeze(W(f_idx, :, comp_idx)); 
            if isrow(w), w = w'; end 

            % Самый ресурсозатратный блок: вычисление огибающих
            for ep_idx = 1:n_epochs
                for tr_idx = 1:n_trials
                    % w' * Cov * w
                    Envs(ep_idx, tr_idx) = w' * X_covs(:, :, ep_idx, tr_idx) * w;
                end
            end

            % Корреляция между триалами
            inters_c = corr(Envs);
            corr_mask = triu(true(size(inters_c)), 1);
            corrs(f_idx, comp_idx) = mean(inters_c(corr_mask));
>>>>>>> Stashed changes
        end
    end

    Cm = mean(X_covs(:,:,:),3);
    Cm_r = Cm + lambda*eye(size(Cm))*trace(Cm)/size(Cm,1);
    Wm = Cm_r^-0.5;

    D_vec = n_channels * (n_channels + 1) / 2;
    X_covsVecW = zeros(n_epochs, D_vec, n_trials);
    for j=1:n_trials
        for i=1:n_epochs
            X_covsVecW(i,:,j) = cov2upper(Wm * X_covs(:,:,i,j) * Wm')';
        end
    end
    
    [Vc, ~, ~] = corrca(X_covsVecW,lambda);
    
    n_comps = size(Vc, 2);
    z_trials = zeros(n_epochs, n_comps, n_trials);
    raw_var = zeros(n_comps, n_trials);
    
    for j = 1:n_trials
        trial_data = X_covsVecW(:,:,j);
        z_tr = trial_data * Vc; 
        
        raw_var(:, j) = var(z_tr, 0, 1)'; 
        
        z_trials(:,:,j) = (z_tr - mean(z_tr, 1)) ./ std(z_tr, [], 1);
    end

    W = zeros(n_comps, n_channels, n_channels);
    A = zeros(n_comps, n_channels, n_channels);
    
    for comp_i = 1:10
        z_trials_comp = squeeze(z_trials(:, comp_i, :));
        [w, a] = my_spoc(X_covs, z_trials_comp, lambda);
        
        W(comp_i,:,:) = w;
        A(comp_i,:,:) = a;
    end
end

<<<<<<< Updated upstream
=======
% =========================================================================

>>>>>>> Stashed changes
