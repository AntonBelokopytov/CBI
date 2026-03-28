function [U, Env_sum, Sources, L_mean, D_mean] = find_induced_envelope(X_all, Fs, Wsize, Ssize, N_comp)
% Поиск индуцированной огибающей с использованием усреднения вероятностных
% лапласианов по нескольким субъектам.
%
% Вход:
% X_all - данные [Nsens x Nsamples x Nsubj]
% Fs    - частота дискретизации
% Wsize - размер окна в секундах
% Ssize - шаг смещения окна в секундах
% N_comp - количество извлекаемых компонент
%
% Выход:
% U       - собственные векторы усредненного Лапласиана
% Env_sum - восстановленная огибающая (сумма мощностей фильтров)
% Sources - временные ряды извлеченных компонент
% L_mean, D_mean - усредненные матрицы Лапласиана и диагоналей

if nargin < 5
    N_comp = 5;
end

[Nsens, Nsamples, Nsubj] = size(X_all);

% 1. Эпохирование и вычисление ковариационных матриц для каждого субъекта
Covs_all_subj = {};
Covs_vec_all_subj = {};
Tcovs_all_subj = {};

% Определим размеры эпох
window_length = round(Wsize * Fs);
step_length = round(Ssize * Fs);
N_epochs = floor((Nsamples - window_length) / step_length) + 1;

fprintf('Разбиение на эпохи: %d окон на субъекта.\n', N_epochs);

% Цикл по субъектам для получения ковариационных матриц и Лапласианов
L_sum = 0;
D_sum = 0;

for subj_idx = 1:Nsubj
    X_subj = X_all(:,:,subj_idx);

    % SVD/PCA для снижения размерности сенсорного пространства (опционально, но полезно для стабильности)
    [Upca, S_pca, ~] = svd(X_subj, 'econ');
    S_pca = diag(S_pca);
    tol = max(size(X_subj)) * eps(S_pca(1));
    r = sum(S_pca > tol);
    ve = S_pca.^2;
    var_explained = cumsum(ve) / sum(ve);
    var_explained(end) = 1;
    n_components = find(var_explained >= 0.99, 1);
    n_components = max(min(n_components, r), 1);
    Upca = Upca(:, 1:n_components);
    X_subj_pca = Upca' * X_subj;

    % Нарезаем на окна
    ep_wins = zeros(n_components, window_length, N_epochs);
    for i = 1:N_epochs
        start_idx = (i-1)*step_length + 1;
        end_idx = start_idx + window_length - 1;
        ep_wins(:,:,i) = X_subj_pca(:, start_idx:end_idx);
    end

    % Ковариационные матрицы
    Covs = zeros(n_components, n_components, N_epochs);
    Covs_vec = zeros(N_epochs, n_components * (n_components + 1) / 2);

    for i = 1:N_epochs
        C = cov(ep_wins(:,:,i)');
        Covs(:,:,i) = C;
        % Верхнетреугольная часть
        Covs_vec(i,:) = C(triu(true(size(C))));
    end

    % Для графа используем Евклидово расстояние между векторами ковариаций (или касательное пространство)
    % Так как у каждого субъекта свое PCA пространство, риманово среднее считать между субъектами сложно
    % Но мы строим граф (L, D) по эпохам ВНУТРИ субъекта!

    % Построение Лапласиана для субъекта (аналогично laplace_embedding.m)
    sigma = 10; % Параметры можно вынести в аргументы
    N_neigb = 10;

    % [L, D, ~, ~] = laplace_embedding(Covs_vec, sigma, N_neigb);
    % Реализуем локально, чтобы не зависеть от внешних файлов
    [idx, D_knn] = knnsearch(Covs_vec, Covs_vec, 'K', N_neigb + 1, 'Distance', 'euclidean');
    idx(:, 1) = [];
    D_knn(:, 1) = [];
    weights = exp(-(D_knn.^2) / (2 * sigma^2));
    row_idx = repmat((1:N_epochs)', 1, N_neigb);
    W_n = sparse(row_idx(:), idx(:), weights(:), N_epochs, N_epochs);
    W = max(W_n, W_n');
    D = diag(sum(W, 2));
    L = D - W;

    % Суммируем Лапласианы
    if subj_idx == 1
        L_sum = L;
        D_sum = D;
    else
        L_sum = L_sum + L;
        D_sum = D_sum + D;
    end

    % Сохраняем данные для восстановления огибающей
    Covs_all_subj{subj_idx} = Covs;
    Covs_vec_all_subj{subj_idx} = Covs_vec;
end

% Усредняем Лапласианы
L_mean = L_sum / Nsubj;
D_mean = D_sum / Nsubj;

% В случае, если матрица D_mean становится вырожденной или почти вырожденной
% из-за разреженности или других факторов, добавим регуляризацию
D_mean_reg = D_mean + eye(size(D_mean)) * 1e-3;

% В Octave eigs с generalized eigenvalue problem иногда багует для маленьких значений ('sm').
% Поэтому мы можем решить эту задачу через eig для полной матрицы:
% L*v = lambda*D*v => (D^-0.5 * L * D^-0.5)*u = lambda*u
D_inv_sqrt = diag(1 ./ sqrt(diag(D_mean_reg)));
L_sym = D_inv_sqrt * L_mean * D_inv_sqrt;

% L_sym - симметричная положительно полуопределенная
% 2. Находим собственные векторы усредненного графа Лапласиана
% 'sm' иногда выдает ошибки нулевого вектора в Octave, используем полные собственные значения для надежности:
[U_sym, S_eig] = eig(full(L_sym));
[S_eig_vals, idx] = sort(diag(S_eig), 'ascend');
U_sym = U_sym(:, idx(1:N_comp + 1));
S_eig = S_eig_vals(1:N_comp + 1);
U = D_inv_sqrt * U_sym; % возвращаемся в исходное пространство
U = U(:, 2:end); % Отбрасываем первую компоненту (соответствует собственному значению 0)

Sources = U;

% 3. Восстановление огибающей для первого собственного вектора
% Мы используем данные первого субъекта для демонстрации (т.к. у каждого свое PCA)
% В идеале, проекция должна быть усреднена или применена к общему сенсорному пространству
comp_idx = 1;
subj_idx_for_env = 1;

Covs_pca = Covs_all_subj{subj_idx_for_env};
Covs_vec_pca = Covs_vec_all_subj{subj_idx_for_env};
n_comp_pca = size(Covs_pca, 1);

% Поскольку U содержит временную динамику по окнам (собственные векторы Лапласиана),
% мы можем напрямую использовать ее как огибающую, как в laplacian_decomposition.m
Env_sum = Sources(:, comp_idx);
Env_sum = (Env_sum - mean(Env_sum)) / std(Env_sum);

end
