function [W, A, X_covs, z] = env_pca(X, ...
Fs, Wsize, Ssize, n_plot_comp)

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
X_covs = [];
for i=1:n_epochs
    for j=1:n_trials
        X_covs(:,:,i,j) = cov(X_epochs(:,:,i,j));
    end
end
mX_covs = mean(X_covs,4);
Cm = riemann_mean(mX_covs);
Wm = Cm^-0.5;

X_covsVecW = [];
for i=1:n_epochs
    for j=1:n_trials
        X_covsVecW(:,i,j) = cov2upper(Wm*X_covs(:,:,i,j)*Wm');
    end
end

X_covsVecWm = mean(X_covsVecW,3);
X_covsVecWm = X_covsVecWm - mean(X_covsVecWm,2);
[Uc,~,~] = svd(X_covsVecWm,'econ');

z = Uc' * X_covsVecWm; z = (z - mean(z,2)) ./ std(z,[],2); z = z';
Af = X_covsVecWm * z;

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

% =========================================================================

function Feat = Tangent_space(COV,C)

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
