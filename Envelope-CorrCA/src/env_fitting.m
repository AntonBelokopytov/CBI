function [W, A, eigenvalues, Epochs_cov, z] = env_fitting(X, ...
Fs, Wsize, Ssize)

[~, n_ch, n_trials] = size(X);

for tr_idx=1:n_trials
    X(:,:,tr_idx) = X(:,:,tr_idx) - mean(X(:,:,tr_idx),1);
end
Xmean = mean(X,3);

X_epochs = [];
for tr_idx=1:n_trials
    mX = X(:,:,tr_idx) - Xmean;
    mX = mX ./ sqrt(trace(cov(mX)));
    X_epochs(:,:,:,tr_idx) = epoch_data(mX,Fs,Wsize,Ssize);
end

[~, ~, n_epochs, ~] = size(X_epochs);
Epochs_cov = [];
for i=1:n_epochs
    for j=1:n_trials
        C = cov(X_epochs(:,:,i,j));
        Epochs_cov(:,:,i,j) = C;
    end
end

for i=1:n_trials
    i
    Dists(:,:,i) = compute_dists(Epochs_cov(:,:,:,i));
end

N_neigb = fix(n_epochs/2); temp_window = 1;
[L, D] = compute_fuzzy_set(mean(Dists,3), N_neigb, temp_window);

[V, S] = eig(L,D);
S = diag(S); [S, idx] = sort(S,'ascend'); V = V(:,idx);
threshold = 1e-10;
valid_idx = S > threshold; valid_idx(1) = false;
V = V(:,valid_idx);
plot(V(:,1))

Cm = riemann_mean(mX_covs);
Wm = Cm^-0.5;
mX_covsW = [];
for i=1:size(mX_covs,3)
    mX_covsW(:,:,i) = Wm * mX_covs(:,:,i) * Wm';
end

N_neigb = fix(n_epochs/4);
[W, W_n] = laplace_embedding(mX_covsW, N_neigb);
D = diag(sum(W_n, 2)); 
L = D - W_n;

Covs_vecW = []; Covs_vec = [];
for i=1:size(mX_covs,3)
    Covs_vecW(:,i) = cov2upper(mX_covsW(:,:,i));
end
Covs_vecW = Covs_vecW - mean(Covs_vecW,2);

[Uc, ~, ~] = svd(Covs_vecW, 'econ');
mX_covsW_pca = Uc' * Covs_vecW;
mX_covsW_pca = mX_covsW_pca - mean(mX_covsW_pca,1);

vLv = mX_covsW_pca * L * mX_covsW_pca'; 
vDv = mX_covsW_pca * D * mX_covsW_pca'; 

[V, S] = eig(vLv,vDv);
S = diag(S); [S, idx] = sort(S,'ascend'); V = V(:,idx);
threshold = 10e-10;
valid_idx = S > threshold;
V = V(:,valid_idx);

z = V' * mX_covsW_pca;
z = (z - mean(z,2)) ./ std(z,[],2);
z = z';

% [V, S] = eig(L,D);
% S = diag(S); [S, idx] = sort(S,'ascend'); V = V(:,idx);
% threshold = 0;
% valid_idx = S > threshold; valid_idx(1) = false;
% V = V(:,valid_idx);
% z = V; z = (z - mean(z,1)) ./ std(z,[],1);
 
Af = Covs_vecW * z; 

n_z = size(z, 2);
n_comp = n_ch; 

W_total = zeros(n_z, n_ch, n_comp);
A_total = zeros(n_z, n_ch, n_comp);
Eig_total = zeros(n_z, n_comp);

for i = 1:n_z
    WW = upper2cov(Af(:, i)); 
    
    [Uw, Sw] = eig(WW); 
    [eig_vals, idx] = sort(diag(Sw), 'descend');
    Uw = Uw(:, idx);
    
    W_curr = Wm * Uw; 
    
    for comp_idx = 1:n_comp
        w_tmp = W_curr(:, comp_idx);
        w_norm = w_tmp / sqrt(w_tmp' * Cm * w_tmp);
        
        W_total(i, :, comp_idx) = w_norm;
        A_total(i, :, comp_idx) = Cm * w_norm; 
    end
    
    Eig_total(i, :) = eig_vals;
end

W = W_total;
A = A_total;
eigenvalues = Eig_total;

visualize(z, eigenvalues, 4, Wsize, Ssize)

end

% =========================================================================

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

% =========================================================================

function [v] = cov2upper(C)
    upper_triu_mask = triu(true(size(C)),1);
    upper_mask = triu(true(size(C)));
    C(upper_triu_mask) = C(upper_triu_mask)*sqrt(2);
    upper_triangle = C(upper_mask);
    v = upper_triangle(:);
end

% =========================================================================

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

% A = riemann_mean(B,epsilon,tol)
%
% Calcul du barycentre des matrice de covariances.
% A : baricentre des K matrices NxN
%
% B : Matrice NxNxK
% epsilon : Pas de la descente de gradient
% tol : arret de la descente si le crit�re < tol

% =========================================================================

function [A, critere, niter] = riemann_mean(B,args)

N_itermax = 100;
if (nargin<2)||(isempty(args))
    tol = 10^-2; %-5
    A = mean(B,3);
else
    tol = args{1};
    A = args{2};
end

niter = 0;
fc = 0;

while (niter<N_itermax)
    niter = niter+1;
    % Tangent space mapping
    T = Tangent_space(B,A);
    % sum of the squared distance
    fcn = sum(sum(T.^2));
    % improvement
    conv = abs((fcn-fc)/fc);
    if conv<tol % break if the improvement is below the tolerance
       break; 
    end
    % arithmetic mean in tangent space
    TA = mean(T,2);
    % back to the manifold
    A = UnTangent_space(TA,A);
    fc = fcn;
end

if niter==N_itermax
    disp('Warning : Nombre d''iterations maximum atteint');
end

critere = fc;

end

% =========================================================================
% 
% function [L, D, W_n, W] = laplace_embedding(Covs, N_neigb)
%     n = size(Covs,3);
%     Dists = zeros(n);
%     for i=1:n-1
%         for j=i+1:n
%             A = Covs(:,:,i);
%             B = Covs(:,:,j);
%             d = distance_riemann(A,B);
%             Dists(i,j) = d;
%         end
%     end
%     Dists = (Dists + Dists');
% 
%     upper_tri_dists = Dists(triu(true(n), 1));
% 
%     sigma = median(upper_tri_dists);
%     if sigma == 0
%         sigma = 1e-5;
%     end
% 
%     W = exp(-(Dists.^2) / (2 * sigma^2));
%     W = W - diag(diag(W));
% 
%     n_epochs = size(Dists, 1);
%     W_n = zeros(n_epochs, n_epochs);
% 
%     % --- k-NN граф ---
%     for i = 1:n_epochs
%         [mvals, mids] = sort(W(i,:), 'descend');
%         W_n(i, mids(1:N_neigb)) = mvals(1:N_neigb);
%     end
% 
%     n_seq = 5;  
%     for i = 1:n_epochs
%         neighbors = (i - n_seq):(i + n_seq);
%         neighbors(neighbors == i) = [];  
%         neighbors = neighbors(neighbors >= 1 & neighbors <= n_epochs);
% 
%         for j = neighbors
%             W_n(i, j) = W(i, j);
%         end
%     end    
%     W_n = (W_n + W_n') / 2; 
% 
%     % --- лапласиан ---
%     D = diag(sum(W_n, 2)); 
%     L = D - W_n;
% end

% =========================================================================

function Dists = compute_dists(Covs)
    n = size(Covs,3);
    Dists = zeros(n);
    
    % 1. Расчет полной матрицы Римановых расстояний
    for i=1:n-1
        for j=i+1:n
            A = Covs(:,:,i);
            B = Covs(:,:,j);
            % d = distance_riemann(A,B); 
            d = norm(A - B); 
            Dists(i,j) = d;
        end
    end
    Dists = (Dists + Dists');

end

% =========================================================================

function [L, D, W] = compute_fuzzy_set(Dists, N_neigb, temp_window)
    % Если размер временного окна не передан, ставим 5 по умолчанию
    if nargin < 3
        temp_window = 5; 
    end
    
    % ИСПРАВЛЕНИЕ: берем первую размерность матрицы расстояний
    n = size(Dists, 1);
    
    W_directed = zeros(n, n);
    
    % Защита от N_neigb < 2
    target_k = log2(max(2, N_neigb)); 
    
    % 2. Построение графа по данным (UMAP Fuzzy Logic)
    for i = 1:n
        distances_i = Dists(i, :);
        sorted_dists = sort(distances_i, 'ascend');
        
        if n > 1
            rho_i = sorted_dists(2);
        else
            rho_i = 0;
        end
        
        sigma_i = 1.0;
        min_sigma = 0.0;
        max_sigma = Inf;
        
        other_idx = [1:i-1, i+1:n];
        dists_to_others = distances_i(other_idx);
        
        for iter = 1:64 
            vals = exp(-max(0, dists_to_others - rho_i) / sigma_i);
            sum_vals = sum(vals);
            
            if abs(sum_vals - target_k) < 1e-5
                break;
            end
            
            % Безопасный бинарный поиск
            if sum_vals > target_k
                max_sigma = sigma_i;
                if min_sigma == 0.0
                    sigma_i = sigma_i / 2.0;
                else
                    sigma_i = (sigma_i + min_sigma) / 2.0;
                end
            else
                min_sigma = sigma_i;
                if isinf(max_sigma)
                    sigma_i = sigma_i * 2.0;
                else
                    sigma_i = (sigma_i + max_sigma) / 2.0;
                end
            end
        end
        W_directed(i, other_idx) = exp(-max(0, dists_to_others - rho_i) / sigma_i);
    end
    
    % 3. Симметризация графа данных (Fuzzy Set Union)
    W_data = W_directed + W_directed' - (W_directed .* W_directed');
    W_data = W_data - diag(diag(W_data));
    
    % 4. Создание графа временных связей
    W_time = zeros(n, n);
    for i = 1:n
        idx_min = max(1, i - temp_window);
        idx_max = min(n, i + temp_window);
        W_time(i, idx_min:idx_max) = 1;
    end
    W_time = W_time - diag(diag(W_time)); 
    
    % 5. Объединение пространственных и временных графов (Fuzzy Set Union)
    W = W_data + W_time - (W_data .* W_time);
    
    % 6. Расчет матриц для Лапласиана
    D = diag(sum(W, 2));
    L = D - W;
end

% =========================================================================

function a = distance_riemann(A,B)

a = sqrt(sum(log(eig(A,B)).^2));

end

% =========================================================================

function [Feat C] = Tangent_space(COV,C)

NTrial = size(COV,3);
N_elec = size(COV,1);
Feat = zeros(N_elec*(N_elec+1)/2,NTrial);

if nargin<2
    C = riemann_mean(COV);
end

index = reshape(triu(ones(N_elec)),N_elec*N_elec,1)==1;
Pinv_sqrt = C^-0.5;
% Psqrt = C^0.5;

for i=1:NTrial
    Tn = logm(Pinv_sqrt*COV(:,:,i)*Pinv_sqrt);
    % Tn = Pinv_sqrt*COV(:,:,i)*Pinv_sqrt;
    % Tn = logm(COV(:,:,i));
    tmp = reshape(sqrt(2)*triu(Tn,1)+diag(diag(Tn)),N_elec*N_elec,1);
    Feat(:,i) = tmp(index);
end

end

% =========================================================================

function COV = UnTangent_space(T,C)
NTrial = size(T,2);
N_elec = (sqrt(1+8*size(T,1))-1)/2;
COV = zeros(N_elec,N_elec,NTrial);

if nargin<2
    C = riemann_mean(COV);
end

index = reshape(triu(ones(N_elec)),N_elec*N_elec,1)==0;

Out = zeros(N_elec*N_elec,NTrial);

Out(not(index),:) = T;
P = C^0.5;
for i=1:NTrial
  tmp = reshape(Out(:,i),N_elec,N_elec,[]);
  tmp = diag(diag(tmp))+triu(tmp,1)/sqrt(2) + triu(tmp,1)'/sqrt(2);
  tmp = P*tmp*P;
  COV(:,:,i) = RiemannExpMap(C,tmp);
end

end

% =========================================================================

function visualize(z, eigenvalues, n_comp, Wsize, Ssize)

[n_epochs, n_z] = size(z);
n_ch = size(eigenvalues,2);

n_plot = min(n_comp, n_z); 
if n_plot > 0
    figure('Name', 'Env-CorrCA: Laplacian Envelopes and Spatial Patterns', ...
           'Position', [100, 100, 1000, 800]);
    
    t_z = (0:n_epochs-1) * Ssize + (Wsize / 2);
    
    for i = 1:n_plot
        subplot(n_plot, 2, 2*i - 1);
        plot(t_z, z(:, i), 'LineWidth', 1.5, 'Color', [0.2 0.4 0.8]);
        title(sprintf('Laplacian Envelope z_{%d}', i));
        xlabel('Time (s)');
        xlim([t_z(1), t_z(end)]);
        grid on;
        
        subplot(n_plot, 2, 2*i);
        stem(1:n_ch, eigenvalues(i,:), 'filled', 'LineWidth', 1.2, 'Color', [0 0 0.8]);
        title(sprintf('SPoC eigenvalues z_{%d}', i));
        xlabel('Channel Index');
        ylabel('Weight');
        xlim([0, n_ch + 1]);
        grid on;
    end
end

end
% =========================================================================
