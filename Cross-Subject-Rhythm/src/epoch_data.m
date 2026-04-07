function X_epo = epoch_data(X, Fs, Ws, Ss)
    W = fix(Ws*Fs);
    S = fix(Ss*Fs);
    n_epochs = floor((size(X,1) - W) / S) + 1; 
    X_epo = zeros(W, size(X,2), n_epochs); 
    
    range = 1:W; 
    for ep = 1:n_epochs
        X_epo(:,:,ep) = X(range,:); 
        range = range + S; 
    end
end
