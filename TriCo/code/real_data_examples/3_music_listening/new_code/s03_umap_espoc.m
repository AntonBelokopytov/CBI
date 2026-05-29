% =====================================================================
% UMAP EMBEDDING
% =====================================================================
u = UMAP("n_neighbors", 20, "n_components", 3, "min_dist", 0);
u.metric = 'euclidean';
u.target_metric = 'euclidean';
R = u.fit_transform(Tcovs(:,ep_mask)');
Rmean = R - mean(R,1);

% =====================================================================
% eSPoC COMPUTATION
% =====================================================================
[W, A, Vf, Vz, corrs, VecCov, Epochs_cov, eigenvalues] = espoc(X_epo(:,:,ep_mask), R');

% Canonical projections calculation
gl_src = Vf' * VecCov;
emb_can_pr = Vz' * R';

