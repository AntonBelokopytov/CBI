function [U, Env_sum, Sources, L_mean, D_mean] = find_induced_envelope(X_all, Fs, Wsize, Ssize, N_comp)
if nargin < 5
    N_comp = 5;
end

[Nsens, Nsamples, Nsubj] = size(X_all);

lambda = 0.01;

sigma = 100;
Dists = [];
for subj_idx = 1:Nsubj
    subj_idx
    Epochs = epoch_data(X_all(:,:,subj_idx)',Fs,Wsize,Ssize);
    
    [~,~,Nepochs] = size(Epochs);

    Covs_reg = zeros(Nsens,Nsens,Nepochs);
    for ep_idx = 1:Nepochs
        C = cov(Epochs(:,:,ep_idx));
        Covs_reg(:,:,ep_idx) = C + lambda * eye(Nsens); 
    end    

    Dist = calc_riemann_dists(Covs_reg);    
    W = exp(-(Dists.^2) / (2 * sigma^2));
    Dists(:,:,subj_idx) = W; 
end

[U, S_eig] = eigs(L, D, N_comp + 1, 'sm');
S_eig = diag(S_eig);
U = U(:, 2:end);

Sources = U;

end

function Dists = calc_riemann_dists(Covs)
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
end

function a = distance_riemann(A,B)
    a = sqrt(sum(log(eig(A,B)).^2));
end
