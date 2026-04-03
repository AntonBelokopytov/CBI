function [W, Rw, Rb] = corrca(X,lambda)
    [T, D, N] = size(X);
    Xc = X - mean(X,1); % Центрирование
    
    Rw = zeros(D,D);
    
    % Векторизованный подсчет для Rb
    SumX = sum(Xc, 3); 
    TotalCov = (SumX' * SumX) / (T-1); % Матрица ковариации суммы всех трайлов
    
    % Считаем только Rw (один цикл)
    for i = 1:N
        Xi = Xc(:,:,i); % squeeze здесь не обязателен для 2D среза
        Rw = Rw + (Xi' * Xi) / (T-1);
    end
    
    % Вычисляем Rb без двойного цикла!
    Rb = (TotalCov - Rw) / (N*(N-1)); 
    Rw = Rw / N;
    
    % shrinkage regularization
    Rw_reg = (1-lambda)*Rw + lambda*mean(eig(Rw))*eye(D);
    
    % generalized eigenvalue problem
    [W, S] = eig(Rb, Rw_reg, 'chol');
    
    % сортировка
    [S, indx] = sort(diag(S), 'descend');
    W = W(:, indx);
end
