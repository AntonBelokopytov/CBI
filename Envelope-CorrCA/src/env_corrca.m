function [W, A, corrs, X_covs, z] = env_corrca(X, ...
Fs, Wsize, Ssize, n_plot_comp)

[~, ~, n_trials] = size(X);

for tr_idx=1:n_trials
    X(:,:,tr_idx) = X(:,:,tr_idx) - mean(X(:,:,tr_idx),1);
end
Xmean = mean(X,3);

% git change

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
Cm = riemann_mean(mX_covs);
Wm = Cm^-0.5;

X_covsVecW = [];
for i=1:n_epochs
    for j=1:n_trials
        X_covsVecW(i,:,j) = cov2upper(Wm*X_covs(:,:,i,j)*Wm');
    end
end

% X_covsVecWm = mean(X_covsVecW,3);
% X_covsVecWm = X_covsVecWm - mean(X_covsVecWm,2);
% [Uc,~,~] = svd(X_covsVecWm,'econ');
% 
% X_covsVecWdr = [];
% for i=1:n_trials
%     X_covsVecWdr(:,:,i) = (Uc' * X_covsVecW(:,:,i))';
% end

% z = Uc' * X_covsVecWm; z = z';

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

corrs = intertr_corrs(W, X_covs, n_plot_comp);

visualize(z, eigenvals, corrs, n_plot_comp, Wsize, Ssize)

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
        end
    end
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function visualize(z, eigenvalues, corrs, n_comp, Wsize, Ssize)
    % z: [n_epochs, n_z] — огибающие
    % eigenvalues: [n_filters, n_components] — собственные значения
    % corrs: [n_filters, n_components] — средние корреляции
    
    [n_epochs, n_z] = size(z);
    [n_filters, n_ch] = size(eigenvalues); 
    n_plot = min(n_comp, n_z); 
    
    if n_plot > 0
        figure('Name', 'Env-CorrCA Analysis', 'Color', 'w', ...
               'Position', [100, 100, 1100, 200 * n_plot + 100]); % Адаптивная высота
        
        t_z = (0:n_epochs-1) * Ssize + (Wsize / 2);
        
        for i = 1:n_plot
            % --- Левая колонка: Огибающие ---
            subplot(n_plot, 2, 2*i - 1);
            plot(t_z, z(:, i), 'LineWidth', 1.3, 'Color', [0.2 0.4 0.8]);
            ylabel(sprintf('z_{%d}', i)); % Короткая метка слева
            xlim([t_z(1), t_z(end)]);
            grid on;
            
            if i == 1, title('Component Envelopes'); end
            
            % Подписываем X только для самого нижнего графика
            if i == n_plot
                xlabel('Time (s)');
            else
                set(gca, 'XTickLabel', []); 
            end
            
            % --- Правая колонка: Собственные значения + Корреляции ---
            subplot(n_plot, 2, 2*i);
            
            % Собственные значения (левая ось)
            yyaxis left
            stem(1:n_ch, eigenvalues(i, :), 'filled', 'MarkerSize', 3.5, 'Color', [0.1 0.1 0.7]);
            ylabel('Eig');
            set(gca, 'YColor', [0.1 0.1 0.7]);
            
            % Корреляции (правая ось)
            yyaxis right
            plot(1:n_ch, corrs(i, :), '-o', 'LineWidth', 1.1, 'MarkerSize', 4, 'Color', [0.7 0.1 0.1]);
            ylabel('Corr');
            set(gca, 'YColor', [0.7 0.1 0.1]);
            ylim([min(0, min(corrs(i,:))), 1.05]); 
            
            xlim([0.5, n_ch + 0.5]);
            grid on;
            
            if i == 1, title('Eigvals & Inter-trial Corrs'); end
            
            if i == n_plot
                xlabel('Component Index');
            else
                set(gca, 'XTickLabel', []);
            end
        end
    end
end

% =========================================================================

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% function [W, A, S] = eig_dec(Epochs_covW, Wm, Cxx)
% 
% gamma = 0.001;
% 
% [~, n_channels, n_epochs, n_trials] = size(Epochs_covW);
% 
% for sub_idx=1:n_trials
%     tr = trace(mean(Epochs_covW(:,:,:,sub_idx),3));
%     for i=1:n_epochs
%         Ci = Epochs_covW(:,:,i,sub_idx);
%         Epochs_covW(:,:,i,sub_idx) = Ci / tr;
%     end
% 
%     m = mean(Epochs_covW(:,:,:,sub_idx),3);
%     for i=1:n_epochs
%         Ci = Epochs_covW(:,:,i,sub_idx);
%         Epochs_covW(:,:,i,sub_idx) = Ci - m;
%     end
% end
% 
% Rw = zeros(n_channels,n_channels);
% Rb = zeros(n_channels,n_channels);
% for ep_idx=1:n_epochs
%     for i=1:n_trials
%         Ci = Epochs_covW(:,:,ep_idx,i);
%         Rw = Rw+Ci*Ci;
%         for j=i+1:n_trials
%             Cj = Epochs_covW(:,:,ep_idx,j); 
%             Rb = Rb+Ci*Cj;
%         end
%     end
% end
% Rw = (Rw + Rw') / 2;
% Rb = (Rb + Rb') / 2;
% 
% Rw = Rw/n_trials/n_epochs;
% Rb = Rb/(n_trials*(n_trials-1))/n_epochs;
% 
% Rw = (1-gamma)*Rw + gamma*mean(eig(Rw))*eye(size(Rw));
% 
% [w,S]=eig(Rb,Rw); [S,indx]=sort(diag(S),'descend'); w=w(:,indx);
% Wpr = Wm*w;
% 
% for comp_idx=1:size(Wpr,2)
%     Wprn = Wpr(:,comp_idx) / sqrt(Wpr(:,comp_idx)' * Cxx * Wpr(:,comp_idx));
%     W(:,comp_idx) = Wprn;
%     A(:,comp_idx) = Cxx * Wprn / (Wprn' * Cxx * Wprn);
% end
% 
% end
