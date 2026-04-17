function [W, A, S] = cspspoc(X_epochs, z, lambda, k)
    % Инициализация параметров по умолчанию
    if nargin < 3
        lambda = 1e-5; 
    end
    if nargin < 4
        k = size(X_epochs, 3) - 1; 
    end
    
    % Стандартизация целевой переменной
    z = (z - mean(z)) ./ std(z);
    [~, n_channels, n_epochs] = size(X_epochs);
    k = min(k, n_epochs - 1);
    
    X_covs = zeros(n_channels, n_channels, n_epochs);
    X_covs_reg = zeros(n_channels, n_channels, n_epochs);
    
    % Вычисление и регуляризация ковариационных матриц
    for ep_i = 1:n_epochs
        C = cov(X_epochs(:,:,ep_i));
        X_covs(:,:,ep_i) = C;
        
        C_reg = C + lambda * (trace(C) / n_channels) * eye(n_channels);
        X_covs_reg(:,:,ep_i) = (C_reg + C_reg') / 2; 
    end
    
    % Вычисление глобального Риманова среднего
    Cm = riemann_mean(X_covs_reg); 
    
    % Предрасчет обратных квадратных корней для локального "отбеливания"
    C_inv_half = zeros(n_channels, n_channels, n_epochs);
    for ep = 1:n_epochs
        C_inv_half(:,:,ep) = X_covs_reg(:,:,ep)^(-1/2); 
    end
    
    % Вычисление попарных расстояний (AIRM)
    Dists = zeros(n_epochs, n_epochs);
    parfor ep_j = 1:n_epochs
        Cj_whiten = C_inv_half(:,:,ep_j); 
        dist_col = zeros(n_epochs, 1);
        
        for ep_i = (ep_j+1):n_epochs
            Ci = X_covs_reg(:,:,ep_i);
            M = Cj_whiten * Ci * Cj_whiten;
            M = (M + M') / 2; 
            
            e = eig(M);
            dist_col(ep_i) = sqrt(sum(log(e).^2));
        end
        Dists(:, ep_j) = dist_col;
    end
    Dists = Dists + Dists';
    
    % Переход от расстояний к графовым весам (RBF-ядро)
    t = median(Dists(Dists > 0)); 
    DistsR = exp(-Dists / t);
    
    % Построение симметричной маски K-ближайших соседей
    KNN_mask = false(n_epochs, n_epochs);
    for ep_i = 1:n_epochs
        [~, sorted_idx] = sort(Dists(:, ep_i), 'ascend');
        neighbors = sorted_idx(2:k+1); 
        KNN_mask(neighbors, ep_i) = true;
    end
    KNN_mask = KNN_mask | KNN_mask';
    
    % Сборка целевой матрицы градиентов (Локальный CSP)
    Cgrad = zeros(n_channels);
    for ep_j = 1:n_epochs
        zj = z(ep_j);
        Cj_whiten = C_inv_half(:,:,ep_j); 
        
        local_Cgrad = zeros(n_channels);
        
        % Полный цикл, так как преобразование не симметрично
        for ep_i = 1:n_epochs
            if ep_i == ep_j || ~KNN_mask(ep_i, ep_j)
                continue; 
            end
            
            Ci = X_covs_reg(:,:,ep_i);
            zi = z(ep_i);
            d = DistsR(ep_i, ep_j); % Используем графовый вес близости
            
            if d > 0 
                % Локальная CSP проекция
                M = Cj_whiten * Ci * Cj_whiten;
                M = (M + M') / 2; 
                
                % Взвешивание дельтой z и расстоянием в графе
                local_Cgrad = local_Cgrad + (zj - zi) * M * d;
            end
        end
        
        Cgrad = Cgrad + local_Cgrad;
    end
    
    % Обычное спектральное разложение (без Cm), т.к. Cgrad уже в отбеленном виде
    [V, S_matrix] = eig(Cgrad);
    
    % Сортируем по абсолютному значению, чтобы найти самые сильные связи
    [S, idx] = sort(abs(diag(S_matrix)), 'descend'); 
    V = V(:, idx);
    
    % Проецируем фильтры обратно в пространство исходных сенсоров
    % с помощью глобального среднего
    W = Cm^(-1/2) * V;
    
    % Нормализация пространственных фильтров
    for w_i = 1:size(W,2)
        w = W(:,w_i) / sqrt((W(:,w_i)' * Cm * W(:,w_i)));
        W(:,w_i) = w;
    end
    
    % Вычисление пространственных паттернов для интерпретации (формула Хауфе)
    A = Cm * W / (W'* Cm * W); 
end