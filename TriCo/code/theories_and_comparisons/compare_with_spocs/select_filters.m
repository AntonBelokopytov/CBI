function [idx_filters, corrs] = select_filters(W, Epochs_cov, z_epo)
    n_filters = size(W, 2);
    n_epochs  = size(Epochs_cov, 3);
    n_sources = size(z_epo, 1); 
    
    % z_epo = (z_epo - mean(z_epo,2)) ./ std(z_epo,[],2);
    % z_epo = Vz' * z_epo;

    % --- 1. Вычисление огибающей (Env) ---
    Env = zeros(n_filters, n_epochs);
    for f_idx = 1:n_filters
        for ep_idx = 1:n_epochs
            Env(f_idx, ep_idx) = W(:, f_idx)' * Epochs_cov(:, :, ep_idx) * W(:, f_idx);
        end
    end
    
    % --- 2. Матрица корреляций ---
    % Строки - источники (z_epo), Столбцы - фильтры (Env)
    R = abs(corr(z_epo', Env')); 
    
    % Предварительное выделение памяти для результатов
    idx_sources = zeros(1, n_sources);
    idx_filters = zeros(1, n_sources);
    corrs       = zeros(1, n_sources);
    
    % --- 3. Итеративный поиск уникальных пар ---
    for i = 1:n_sources
        % Находим глобальный максимум в оставшейся матрице R
        [val, ind] = max(R(:));
        [row, col] = ind2sub(size(R), ind);
        
        % Сохраняем результаты (row - это индекс источника, col - индекс фильтра)
        idx_sources(i) = row;
        idx_filters(i) = col;
        corrs(i)       = val;
        
        % Исключаем выбранный фильтр и источник из дальнейшего поиска
        R(:, col) = -inf; 
        R(row, :) = -inf; 
    end
    
    % --- 4. Сортировка результатов ---
    % Упорядочиваем результаты так, чтобы они соответствовали исходному порядку строк в z_epo
    [~, ord] = sort(idx_sources);
    idx_filters = idx_filters(ord);
    corrs       = corrs(ord);
end
