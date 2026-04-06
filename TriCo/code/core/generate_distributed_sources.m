function [X_s, X_bg, X_n, Z, GA, S] = generate_distributed_sources(G, Nsrc, Ndistr, flanker, Ts, Fs)
    N = Ts * Fs;
    flanker = flanker * Fs;
    
    % Установка фильтров
    [b, a] = butter(4, [8, 12] / (Fs / 2)); % Альфа-диапазон для источников
    [bn, an] = butter(4, [1, 35] / (Fs / 2)); % Для шума сенсоров
    [be, ae] = butter(4, 0.5 / (Fs / 2), 'low'); % Для огибающих
    
    Gx = G(:, 1:3:end);  
    Gy = G(:, 2:3:end);  
    Gz = G(:, 3:3:end);  
    [Nsens, Nsites] = size(Gx);
    
    GA = zeros(Nsens, Nsrc);
    src_indsA = randperm(Nsites, Nsrc); 
    
    for i = 1:Nsrc
        src_idx = src_indsA(i);
        r = rand(3, 1) * 2 - 1;
        r = r / norm(r);          
        GA(:, i) = Gx(:, src_idx)*r(1) + Gy(:, src_idx)*r(2) + Gz(:, src_idx)*r(3);
    end
    
    S_full = filtfilt(b, a, randn(Nsrc, N + 2*flanker)')';
    M_full = filtfilt(be, ae, randn(Nsrc, N + 2*flanker)')';
    
    S = zeros(Nsrc, N);
    Z = zeros(Nsrc, N);
    
    for k = 1:Nsrc
        s_k = S_full(k, :);
        s_k = (s_k - mean(s_k)) / std(s_k);
        
        env = abs(hilbert(s_k));
        s_k = s_k ./ (env + eps); 
        
        m_k = M_full(k, :); 
        m_k = (m_k - mean(m_k)) / std(m_k);
        m_k = m_k - min(m_k) + eps; 
        m_k = m_k / mean(m_k);
        
        s_k = s_k .* m_k;
        s_k = (s_k - mean(s_k)) / std(s_k);
        
        S(k, :) = s_k(flanker+1 : end-flanker);
        m_k_cropped = m_k(flanker+1 : end-flanker);
        
        Z(k, :) = m_k_cropped.^2; 
    end
    
    X_s = GA(:, 1:Ndistr) * S(1:Ndistr, :);
    X_bg = GA(:, Ndistr+1:end) * S(Ndistr+1:end, :);
    
    X_n = filtfilt(bn, an, randn(Nsens, N + 2*flanker)')';
    X_n = X_n(:, flanker+1 : end-flanker); 
    X_n = X_n - mean(X_n, 2);
    X_n = X_n ./ std(X_n, 0, 2);
end