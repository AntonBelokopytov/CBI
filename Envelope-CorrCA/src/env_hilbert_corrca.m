function [W, A, corrs, z] = env_hilbert_corrca(X,Fs,Wsize,Ssize,n_plot_comp)
[T, D, n_trials] = size(X);

for tr_idx=1:n_trials
    X(:,:,tr_idx) = X(:,:,tr_idx) - mean(X(:,:,tr_idx),1);
end
Xmean = mean(X,3);
for tr_idx=1:n_trials
    mX = X(:,:,tr_idx) - Xmean;
    mX = mX ./ sqrt(trace(cov(mX)));
    X(:,:,tr_idx) = mX;
end

Xh_full = zeros(T, D, n_trials);
for tr_idx=1:n_trials
    Xh_full(:,:,tr_idx) = hilbert(X(:,:,tr_idx));
end

pad_len = round(Fs/4);
Xh = Xh_full(pad_len+1 : end-pad_len, :, :);

[T_short, ~, ~] = size(Xh);
K = D * (D + 1) / 2;
X_power = zeros(T_short, K, n_trials); 

for tr_idx = 1:n_trials
    Y = Xh(:, :, tr_idx);
    idx = 1;
    for i = 1:D
        for j = i:D
            if i == j
                X_power(:, idx, tr_idx) = Y(:, i) .* conj(Y(:, j));
            else
                X_power(:, idx, tr_idx) = sqrt(2) * Y(:, i) .* conj(Y(:, j));
            end
            idx = idx + 1;
        end
    end
end

mX_power = mean(X_power,3)';
mX_power = mX_power - mean(mX_power,2);
[Uc, ~, ~] = svd(mX_power, 'econ');
top_comp = min(D, size(Uc, 2)); 
Uc = Uc(:, 1:top_comp);

X_powerdr = zeros(T_short, top_comp, n_trials);
for i = 1:n_trials
    X_powerdr(:,:,i) = X_power(:,:,i) * Uc; 
end

[Vc, ~, ~] = corrca(X_powerdr);

z = abs(mean(X_powerdr,3) * Vc);
z = (z - mean(z,1)) ./ std(z,[],1);

X_epochs = [];
for tr_idx=1:n_trials
    X_epochs(:,:,:,tr_idx) = epoch_data(X(pad_len+1:end-pad_len,:,tr_idx),Fs,Wsize,Ssize);
end
Z_epochs = epoch_data(z,Fs,Wsize,Ssize);
Z_epochs = squeeze(mean(Z_epochs,1))';
Z_epochs = (Z_epochs - mean(Z_epochs,1)) ./ std(Z_epochs,[],1);

[~,~,n_epochs,~] = size(X_epochs);

X_covs = [];
for ep_idx=1:n_epochs
    for tr_idx=1:n_trials
        X_covs(:,:,ep_idx,tr_idx) = cov(X_epochs(:,:,ep_idx,tr_idx));
    end
end
mX_covs = mean(X_covs,4);

Cm = riemann_mean(mX_covs);
Wm = Cm^-0.5;

mX_covsVecW = [];
for ep_idx=1:n_epochs
    mX_covsVecW(:,ep_idx) = cov2upper(Wm * mX_covs(:,:,ep_idx) * Wm');
end
mX_covsVecW = mX_covsVecW - mean(mX_covsVecW,2);

Af = mX_covsVecW * Z_epochs;

W = []; A = [];
for i=1:size(Af,2)
    [w, a, s] = project_filters_to_manifold(Af(:,i), Wm, Cm);
    W(i,:,:) = w;
    A(i,:,:) = a;
    eigenvals(i,:) = s;
end

corrs = intertr_corrs(W, X_covs, n_plot_comp);

visualize(Z_epochs, eigenvals, corrs, n_plot_comp, Wsize, Ssize, pad_len, Fs, T)

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
[~, D, N] = size(X);

Rw_sum = zeros(D, D);
for i = 1:N
    Rw_sum = Rw_sum + cov(X(:,:,i));
end
Rw = Rw_sum / N;

X_mean = mean(X, 3);
Rt = (N^2) * cov(X_mean);

% Вычитаем сумму внутриэпохальных ковариаций из общей ковариации
Rb_sum = Rt - Rw_sum;
Rb = Rb_sum / (N * (N - 1)); 

% 3. Shrinkage regularization
Rw_reg = (1-gamma)*Rw + gamma*mean(eig(Rw))*eye(D);

% 4. Generalized eigenvalue problem
[W, S] = eig(Rb, Rw_reg, 'chol');

% 5. Сортировка по убыванию
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

function visualize(z, eigenvalues, corrs, n_comp, Wsize, Ssize, pad_len, Fs, T)
    % z: [n_epochs, n_z] — огибающие
    % eigenvalues: [n_filters, n_components] — собственные значения
    % corrs: [n_filters, n_components] — средние корреляции
    
    [n_epochs, n_z] = size(z);
    [n_filters, n_ch] = size(eigenvalues); 
    n_plot = min(n_comp, n_z); 
    
    if n_plot > 0
        figure('Name', 'Env-CorrCA Analysis', 'Color', 'w', ...
               'Position', [100, 100, 1100, 200 * n_plot + 100]); 
        
        % Пересчет времени с учетом сдвига на величину паддинга
        pad_time = pad_len / Fs;
        total_time = T / Fs;
        
        % Вектор времени смещается вправо на pad_time
        t_z = pad_time + (0:n_epochs-1) * Ssize + (Wsize / 2);
        
        for i = 1:n_plot
            % --- Левая колонка: Огибающие ---
            ax1 = subplot(n_plot, 2, 2*i - 1);
            hold(ax1, 'on');
            
            % Определяем границы по Y для красивой заливки краев
            yl = [min(z(:,i))*0.95, max(z(:,i))*1.05];
            if yl(1) == yl(2), yl = [yl(1)-1, yl(2)+1]; end
            
            % Отрисовка отброшенных краев (серые прямоугольники)
            fill([0, pad_time, pad_time, 0], [yl(1) yl(1) yl(2) yl(2)], ...
                 [0.9 0.9 0.9], 'EdgeColor', 'none', 'FaceAlpha', 0.5);
            fill([total_time-pad_time, total_time, total_time, total_time-pad_time], ...
                 [yl(1) yl(1) yl(2) yl(2)], [0.9 0.9 0.9], 'EdgeColor', 'none', 'FaceAlpha', 0.5);
            
            % Линии-разделители
            xline(pad_time, '--', 'Color', [0.6 0.6 0.6], 'LineWidth', 1);
            xline(total_time - pad_time, '--', 'Color', [0.6 0.6 0.6], 'LineWidth', 1);
            
            % Сам график огибающей
            plot(t_z, z(:, i), 'LineWidth', 1.5, 'Color', [0.2 0.4 0.8]);
            
            ylabel(sprintf('z_{%d}', i)); 
            xlim([0, total_time]); % Жестко фиксируем ось X на всю длину оригинальной эпохи
            ylim(yl);
            grid on;
            hold(ax1, 'off');
            
            if i == 1, title('Component Envelopes (Gray: Truncated Padding)'); end
            
            if i == n_plot
                xlabel('Time (s)');
            else
                set(gca, 'XTickLabel', []); 
            end
            
            % --- Правая колонка: Собственные значения + Корреляции ---
            subplot(n_plot, 2, 2*i);
            
            yyaxis left
            stem(1:n_ch, eigenvalues(i, :), 'filled', 'MarkerSize', 3.5, 'Color', [0.1 0.1 0.7]);
            ylabel('Eig');
            set(gca, 'YColor', [0.1 0.1 0.7]);
            
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
