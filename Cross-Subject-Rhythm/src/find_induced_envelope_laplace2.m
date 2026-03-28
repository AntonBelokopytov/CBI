function [z, L, D_graph] = find_induced_envelope_laplace2(X_all, Fs, Wsize, Ssize, N_neigb, lambda)
    % Функция поиска консенсусной индуцированной огибающей на основе алгоритма
    % env_laplace_dec2 (Envelope-CorrCA). Адаптирована для межсубъектного
    % анализа (субъекты выступают в роли триалов), не вычисляет пространственные
    % фильтры и паттерны.
    %
    % ВХОД:
    % X_all   - матрица данных [n_channels x n_samples x n_subjects]
    % Fs      - частота дискретизации, Гц
    % Wsize   - размер окна, сек
    % Ssize   - шаг смещения окна, сек
    % N_neigb - количество ближайших соседей (k) для графа k-NN
    % lambda  - коэффициент регуляризации ковариационных матриц
    %
    % ВЫХОД:
    % z       - извлеченные огибающие (собственные векторы Лапласиана),
    %           размер [n_epochs x (n_epochs-1)]
    % L       - усредненная матрица Лапласа
    % D_graph - матрица степеней вершин графа (диагональная)

    if nargin < 5 || isempty(N_neigb), N_neigb = []; end
    if nargin < 6 || isempty(lambda), lambda = 1e-6; end

    [n_ch, n_samples, n_subj] = size(X_all);

    % Центрирование данных по времени для каждого субъекта
    for s_idx = 1:n_subj
        X_all(:,:,s_idx) = X_all(:,:,s_idx) - mean(X_all(:,:,s_idx), 2);
    end

    % Вычисляем общее среднее (по всем субъектам)
    Xmean = mean(X_all, 3);

    % 1. Эпохирование и нормализация
    X_epochs = [];
    for s_idx = 1:n_subj
        mX = X_all(:,:,s_idx) - Xmean;

        % Нормализация на след ковариационной матрицы, чтобы уравнять
        % амплитуды между субъектами
        C_global = cov(mX');
        mX = mX ./ sqrt(trace(C_global));

        % Нарезка на эпохи (транспонируем mX для функции epoch_data,
        % так как epoch_data ожидает [samples x channels])
        X_epo_temp = epoch_data(mX', Fs, Wsize, Ssize);
        % X_epo_temp имеет размер [samples_per_epoch x n_channels x n_epochs]
        % Для удобства переведем в [n_channels x samples_per_epoch x n_epochs x n_subj]
        X_epochs(:,:,:,s_idx) = permute(X_epo_temp, [2, 1, 3]);
    end

    [~, ~, n_epochs, ~] = size(X_epochs);
    if isempty(N_neigb), N_neigb = n_epochs; end
    Epochs_cov = zeros(n_ch, n_ch, n_epochs, n_subj);

    % 2. Считаем регуляризованные ковариации для каждого субъекта
    for i = 1:n_epochs
        for j = 1:n_subj
            % Транспонируем обратно для расчета cov, который ожидает [samples x variables]
            C = cov(X_epochs(:,:,i,j)');
            % Регуляризация СТО (Shrinkage-like)
            C_reg = C + lambda * (trace(C) / n_ch) * eye(n_ch);
            % Симметризация
            C_reg = (C_reg + C_reg') / 2;
            Epochs_cov(:,:,i,j) = C_reg;
        end
    end

    % 3. Построение консенсусного графа Лапласа
    All_W = zeros(n_epochs, n_epochs, n_subj);
    for s_idx = 1:n_subj
        % Считаем Римановы расстояния между эпохами для текущего субъекта
        Subject_Dists = calc_riemann_dists(Epochs_cov(:,:,:,s_idx));
        % Строим матрицу смежности (весов) W для субъекта
        All_W(:,:,s_idx) = build_graph_from_dists(Subject_Dists, N_neigb);
    end

    % Усредняем графы по всем субъектам (консенсусный граф)
    W_graph = mean(All_W, 3);
    D_graph = diag(sum(W_graph, 2));
    L = D_graph - W_graph;

    % 4. Совместная диагонализация (поиск огибающих z)
    % Обобщенная задача на собственные значения: L*v = lambda*D*v
    % Используем eig для полных матриц (работает надежнее eigs в Octave для небольших матриц эпох)
    [V, S] = eig(L, D_graph);
    S = diag(S);
    [S, idx] = sort(S,'ascend');
    V = V(:,idx);

    % Оставляем только значимые (ненулевые) компоненты.
    % Матрица Лапласа всегда имеет одно собственное значение равное 0
    % (или очень близкое к нему), соответствующее константному вектору.
    valid_idx = S > 1e-10;
    valid_idx(1) = false; % Принудительно убираем первый тривиальный вектор

    V = V(:, valid_idx);
    z = V;

    % Стандартизация огибающих (Z-score нормализация по времени)
    z = (z - mean(z, 1)) ./ std(z, [], 1);
end

% =========================================================================
% Вспомогательные функции (взяты из env_laplace_dec2.m)
% =========================================================================

function X_epo = epoch_data(X, Fs, Ws, Ss)
    % Вход X: [samples x channels]
    W = fix(Ws*Fs);
    S = fix(Ss*Fs);
    range = 1:W; ep = 1;
    X_epo = [];
    while range(end) <= size(X,1)
        X_epo(:,:,ep) = X(range,:);
        range = range + S; ep = ep + 1;
    end
end

function Dists = calc_riemann_dists(Covs)
    % Вычисление попарных Римановых расстояний между ковариационными матрицами
    n = size(Covs,3);
    Dists = zeros(n);
    for i=1:n-1
        for j=i+1:n
            A = Covs(:,:,i);
            B = Covs(:,:,j);
            d = distance_riemann(A,B);
            Dists(i,j) = d;
        end
    end
    Dists = (Dists + Dists'); % Симметризация матрицы расстояний
end

function a = distance_riemann(A,B)
    % Риманово расстояние: sqrt(sum(log(eig(A,B)).^2))
    % Функция eig(A,B) решает A*v = lambda*B*v
    eig_vals = eig(A,B);
    % Оставляем только положительные собственные значения во избежание комплексных чисел
    % (хотя для положительно определенных матриц они и так должны быть положительными)
    eig_vals(eig_vals <= 0) = eps;
    a = sqrt(sum(log(eig_vals).^2));
end

function W = build_graph_from_dists(Dists, N_neigb)
    n = size(Dists, 1);
    k = max(1, min(N_neigb, n - 1)); % Исправлено ограничение снизу до 1

    % 1. Находим локальный масштаб sigma для каждой эпохи
    % sigma_i - это расстояние до k-го соседа
    sigmas = zeros(n, 1);
    for i = 1:n
        distances_i = Dists(i, :);
        % Сортируем расстояния по возрастанию
        sorted_dists = sort(distances_i, 'ascend');

        % Теперь вызов (k + 1) абсолютно безопасен, так как первое
        % расстояние всегда 0 (до самой себя)
        sigmas(i) = sorted_dists(k + 1);

        % Защита от нулевого sigma (если есть идентичные эпохи)
        if sigmas(i) < 1e-10
            sigmas(i) = 1e-10;
        end
    end

    % 2. Строим матрицу весов (Self-Tuning Gaussian Kernel)
    W = zeros(n, n);
    for i = 1:n
        for j = i+1:n
            % Возводим расстояние в квадрат для Гауссиана
            d_sq = Dists(i, j)^2;

            % Перемножаем локальные масштабы точек i и j
            scale = sigmas(i) * sigmas(j);

            % Вычисляем вес связи
            w_val = exp(-d_sq / scale);

            W(i, j) = w_val;
            W(j, i) = w_val; % Граф сразу получается симметричным
        end
    end

    % Удаляем диагональ (нет петель в графе)
    W = W - diag(diag(W));
end
