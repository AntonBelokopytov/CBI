function [W, A, eigenvalues, Epochs_cov, z] = env_gradient_dec(X, ...
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
for tr_i=1:n_trials
    for ep_i=1:n_epochs
        Epochs_cov(:,:,ep_i,tr_i) = cov(X_epochs(:,:,ep_i,tr_i));
    end
end

alpha = 10e-10; 
Dists = zeros(n_epochs - 1, n_trials); 
Cij = zeros(n_ch, n_ch, n_epochs - 1, n_trials); 
for tr_i = 1:n_trials
    for ep_i = 1:(n_epochs - 1)
        Ci = Epochs_cov(:,:,ep_i, tr_i);
        Cj = Epochs_cov(:,:,ep_i + 1, tr_i);
        
        Cij(:,:,ep_i,tr_i) = Cj - Ci;

        Ci_reg = Ci + alpha * trace(Ci) * eye(size(Ci, 1));
        Cj_reg = Cj + alpha * trace(Cj) * eye(size(Cj, 1));
        
        Dists(ep_i, tr_i) = distance_riemann(Ci_reg, Cj_reg);
    end
end

DistsM = mean(Dists,2);

for tr_i=1:size(Cij,4)
    for ep_i=1:size(Cij,3)
        Cijd(:,:,ep_i,tr_i) = Cij(:,:,ep_i,tr_i) ./ DistsM(ep_i);
    end
end

Cijdm = mean(Cijd,[3,4]);

mX_covs = mean(Epochs_cov,4);
Cm = mean(mX_covs,3);

[W,S] = eig(Cijdm,Cm); [S,idx] = sort(diag(S),'ascend'); W=W(:,idx);

z = [];
for w_i = 1:38
    w = W(:,w_i);
    for ep_i=1:size(mX_covs,3)
        z(ep_i,w_i) = w' * mX_covs(:,:,ep_i) * w;
    end
end


A = Cm * W;
eigenvalues = S;

visualize(z, eigenvalues, 6, Wsize, Ssize)

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

function a = distance_riemann(A,B)

a = sqrt(sum(log(eig(A,B)).^2));

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
