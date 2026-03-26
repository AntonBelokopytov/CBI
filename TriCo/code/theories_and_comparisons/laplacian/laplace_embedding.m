function [L, D, W_n, W] = laplace_embedding(Covs, sigma, N_neigb)
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
    Dists = (Dists + Dists');

    W = exp(-(Dists.^2) / (2 * sigma^2));
    W = W - diag(diag(W));
    
    n_epochs = size(Dists, 1);
    W_n = zeros(n_epochs, n_epochs);

    for i = 1:n_epochs
        [mvals, mids] = sort(W(i,:), 'descend');
        W_n(i, mids(1:N_neigb)) = mvals(1:N_neigb);
    end

    W_n = (W_n + W_n') / 2;
    
    D = diag(sum(W_n, 2)); 
    L = D - W_n;
end
