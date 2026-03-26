# -*- coding: utf-8 -*-
"""
Created on Fri Feb 20 20:13:22 2026

@author: ansbel
"""

import numpy as np
from scipy.linalg import sqrtm
from scipy.linalg import eigh
# from scipy.signal import butter, filtfilt

#####################################################################

def env_corrca(X, Fs):
    """
    Parameters
    ----------
    X : ndarray, shape (T, n_ch, n_trials)
    Fs : float
    Wsize : float (window size in seconds)
    Ssize : float (step size in seconds)
    method : 'light' or 'full'

    Returns
    -------
    W : ndarray
    A : ndarray
    corrs : ndarray
    Epochs_cov : ndarray
    S : ndarray (eigenvalues)
    """
    method = 'full'
    Wsize = 0.25 # 2/fmin # 2 period step (in seconds)
    Ssize = 0.1 # Wsize/4 # (in seconds)

    # band = [fmin, fmax]
    # b, a = butter(4, np.array(band) / (Fs / 2), btype='band')
    # Xfilt = filtfilt(b, a, X, axis=-1)

    X = np.permute_dims(X,(2,1,0))
    
    opt = {}
    opt["X_min_var_explained"] = 0.99
    opt["whitening_reg"] = 0.01

    T, n_ch, n_trials = X.shape

    # -----------------------------
    # Epoching + normalization
    # -----------------------------
    X_epochs = []

    for tr_idx in range(n_trials):

        Xi = X[:, :, tr_idx]

        # center channels
        Xi = Xi - Xi.mean(axis=0, keepdims=True)

        # normalize by sqrt(trace(cov))
        cov_trace = np.trace(np.cov(Xi.T))
        Xi = Xi / np.sqrt(cov_trace)

        X[:, :, tr_idx] = Xi

        ep = epoch_data(Xi, Fs, Wsize, Ssize)
        X_epochs.append(ep)

    X_epochs = np.stack(X_epochs, axis=-1)
    # shape: (W, n_ch, n_epochs, n_trials)

    _, _, n_epochs, _ = X_epochs.shape

    # -----------------------------
    # Whitening + covariance series
    # -----------------------------
    X_epochs_flat = X_epochs.reshape(X_epochs.shape[0], n_ch, -1)

    VecCov, Wm, Cxx, Epochs_cov, Epochs_covW = \
        get_white_covariance_series(X_epochs_flat, opt)

    Epochs_cov = Epochs_cov.reshape(n_ch, n_ch, n_epochs, n_trials)
    Epochs_covW = Epochs_covW.reshape(n_ch, n_ch, n_epochs, n_trials)

    # -----------------------------
    # Method switch
    # -----------------------------
    if method == 'light': # IN PROGRESS

        W, A, S = eig_dec(Epochs_covW, Wm, Cxx)

    elif method == 'full':

        VecCovdr, Uf = project_to_pc(VecCov, opt["X_min_var_explained"])
        n_feat = VecCovdr.shape[0]

        VecCovdr = VecCovdr.reshape(n_feat, n_epochs, n_trials)
        VecCovdr = np.transpose(VecCovdr, (1, 0, 2))

        Vf, Rw, Rb = corrca(VecCovdr)

        Af = Uf @ Rw @ Vf[:, 0]

        W, A, S = project_filters_to_manifold(Af, Wm, Cxx)

    else:
        raise ValueError("method must be 'light' or 'full'")

    # -----------------------------
    # Inter-trial envelope correlation
    # -----------------------------
    n_components = W.shape[1]
    corrs = np.zeros(n_components)

    for comp_idx in range(n_components):

        Envs = np.zeros((n_epochs, n_trials))

        w = W[:, comp_idx]

        for ep_idx in range(n_epochs):
            for tr_idx in range(n_trials):

                Envs[ep_idx, tr_idx] = (
                    w.T @ Epochs_cov[:, :, ep_idx, tr_idx] @ w
                )

        inters_c = np.corrcoef(Envs.T)

        mask = np.triu(np.ones_like(inters_c, dtype=bool), 1)

        corrs[comp_idx] = inters_c[mask].mean()
    
    return W, A, corrs, Epochs_cov, S

#####################################################################

def epoch_data(X, Fs, Ws, Ss):
    """
    Parameters
    ----------
    X : ndarray, shape (T, n_ch)
    Fs : sampling rate
    Ws : window size (seconds)
    Ss : step size (seconds)

    Returns
    -------
    X_epo : ndarray, shape (W, n_ch, n_epochs)
    """

    W = int(Ws * Fs)
    S = int(Ss * Fs)

    T = X.shape[0]

    epochs = []
    start = 0

    while start + W <= T:
        epochs.append(X[start:start + W, :])
        start += S

    if len(epochs) == 0:
        return np.empty((W, X.shape[1], 0))

    return np.stack(epochs, axis=2)

#####################################################################

def cov2upper(C):
    n = C.shape[0]
    C = C.copy()

    mask_strict = np.triu(np.ones((n, n), dtype=bool), k=1)
    C[mask_strict] *= np.sqrt(2)

    # column-major индексы
    rows, cols = np.where(np.triu(np.ones((n, n)), k=0))
    order = np.lexsort((rows, cols))  # сортировка по столбцу

    rows = rows[order]
    cols = cols[order]

    return C[rows, cols]

##################################################################### 

def upper2cov(v):
    # solve n from n(n+1)/2 = len(v)
    n = int((-1 + np.sqrt(1 + 8 * len(v))) / 2)

    if n * (n + 1) // 2 != len(v):
        raise ValueError("Vector length does not correspond to triangular matrix.")

    C = np.zeros((n, n))

    # column-major order indices (like MATLAB)
    rows, cols = np.where(np.triu(np.ones((n, n)), k=0))
    order = np.lexsort((rows, cols))  # sort by column, then row

    rows = rows[order]
    cols = cols[order]

    # fill upper triangle
    C[rows, cols] = v

    # undo sqrt(2) scaling on strictly upper triangle
    mask_strict = rows < cols
    C[rows[mask_strict], cols[mask_strict]] /= np.sqrt(2)

    # symmetrize
    C = C + np.triu(C, 1).T

    return C

#####################################################################

def get_white_covariance_series(X_epochs, opt):
    """
    Parameters
    ----------
    X_epochs : ndarray, shape (W, n_channels, n_epochs)
    opt : dict with keys:
        - whitening_reg

    Returns
    -------
    VecCov : ndarray, shape (n_features, n_epochs)
    Wm : whitening matrix
    Cxx : mean covariance matrix
    Epochs_cov : ndarray (n_channels, n_channels, n_epochs)
    Epochs_covW : ndarray (n_channels, n_channels, n_epochs)
    """

    _, n_channels, n_epochs = X_epochs.shape

    # number of unique upper-triangular entries
    n_features = n_channels * (n_channels + 1) // 2

    # ---------------------------------
    # Covariance per epoch
    # ---------------------------------
    Epochs_cov = np.zeros((n_channels, n_channels, n_epochs))

    for ep_idx in range(n_epochs):
        # MATLAB cov(X) expects observations in rows
        Xcov = np.cov(X_epochs[:, :, ep_idx].T)
        Epochs_cov[:, :, ep_idx] = Xcov

    # ---------------------------------
    # Mean covariance
    # ---------------------------------
    Cxx = np.mean(Epochs_cov, axis=2)

    # ---------------------------------
    # Whitening matrix
    # ---------------------------------
    reg = opt["whitening_reg"]
    trace_term = np.trace(Cxx) / n_channels

    Cxx_r = Cxx + reg * trace_term * np.eye(n_channels)

    iWm = sqrtm(Cxx_r)
    Wm = np.linalg.inv(iWm)

    # ---------------------------------
    # Whitened covariance series
    # ---------------------------------
    Epochs_covW = np.zeros_like(Epochs_cov)
    VecCov = np.zeros((n_features, n_epochs))

    for ep_idx in range(n_epochs):
        XcovW = Wm @ Epochs_cov[:, :, ep_idx] @ Wm.T
        Epochs_covW[:, :, ep_idx] = XcovW
        VecCov[:, ep_idx] = cov2upper(XcovW)

    return VecCov, Wm, Cxx, Epochs_cov, Epochs_covW

#####################################################################

def project_to_pc(X, min_var_explained):
    """
    PCA-based projection preserving minimum variance explained.

    Parameters
    ----------
    X : ndarray, shape (n_features, n_samples)
    min_var_explained : float

    Returns
    -------
    X_proj : projected data
    U : projection matrix
    """

    if min_var_explained != 1:

        # SVD (econ)
        U, S, Vt = np.linalg.svd(X, full_matrices=False)

        # estimate rank
        tol = max(X.shape) * np.finfo(S.dtype).eps * S[0]
        r = np.sum(S > tol)

        # cumulative variance explained
        ve = S**2
        var_explained = np.cumsum(ve) / np.sum(ve)

        # ensure last entry is exactly 1
        var_explained[-1] = 1.0

        # number of components
        n_components = np.searchsorted(var_explained, min_var_explained) + 1

        # clamp between 1 and r
        n_components = max(min(n_components, r), 1)

        U = U[:, :n_components]

        # project
        X_proj = U.T @ X

    else:
        U = np.eye(X.shape[0])
        X_proj = X.copy()

    return X_proj, U

#####################################################################

def corrca(X):
    """
    CorrCA implementation.

    Parameters
    ----------
    X : ndarray, shape (T, D, N)
        T - time
        D - feature dimension
        N - number of trials / subjects

    Returns
    -------
    W : spatial filters (D, D)
    Rw : within covariance
    Rb : between covariance
    """

    gamma = 1e-5  # shrinkage

    T, D, N = X.shape

    # ---------------------------------
    # Center over time
    # ---------------------------------
    Xc = X - X.mean(axis=0, keepdims=True)

    Rw = np.zeros((D, D))
    Rb = np.zeros((D, D))

    # ---------------------------------
    # Compute covariances
    # ---------------------------------
    for i in range(N):

        Xi = Xc[:, :, i]  # (T × D)

        # within covariance
        Ci = (Xi.T @ Xi) / (T - 1)
        Rw += Ci

        # between covariance
        for j in range(i + 1, N):
            Xj = Xc[:, :, j]
            Cij = (Xi.T @ Xj) / (T - 1)
            Rb += Cij

    # ---------------------------------
    # Normalize
    # ---------------------------------
    Rw = Rw / N
    Rb = (Rb + Rb.T) / (N * (N - 1))

    # ---------------------------------
    # Shrinkage regularization
    # ---------------------------------
    mean_eig = np.mean(np.linalg.eigvalsh(Rw))
    Rw_reg = (1 - gamma) * Rw + gamma * mean_eig * np.eye(D)

    # ---------------------------------
    # Generalized eigenvalue problem
    # Rb w = λ Rw_reg w
    # ---------------------------------
    eigvals, W = eigh(Rb, Rw_reg)

    # sort descending
    idx = np.argsort(eigvals)[::-1]
    W = W[:, idx]

    return W, Rw, Rb

#####################################################################

def eig_dec(Epochs_covW, Wm, Cxx):
    """
    Parameters
    ----------
    Epochs_covW : ndarray (n_ch, n_ch, n_epochs, n_trials)
    Wm : whitening matrix
    Cxx : mean covariance matrix

    Returns
    -------
    W : spatial filters
    A : spatial patterns
    S : eigenvalues
    """

    gamma = 1e-5

    n_ch, _, n_epochs, n_trials = Epochs_covW.shape

    # --------------------------------------------
    # Per-trial normalization + centering
    # --------------------------------------------
    for sub_idx in range(n_trials):

        # mean over epochs
        mean_cov = np.mean(Epochs_covW[:, :, :, sub_idx], axis=2)

        tr = np.trace(mean_cov)

        # normalize each epoch by trace
        for i in range(n_epochs):
            Ci = Epochs_covW[:, :, i, sub_idx]
            Epochs_covW[:, :, i, sub_idx] = Ci / tr

        # recompute mean after normalization
        m = np.mean(Epochs_covW[:, :, :, sub_idx], axis=2)

        # subtract mean
        for i in range(n_epochs):
            Ci = Epochs_covW[:, :, i, sub_idx]
            Epochs_covW[:, :, i, sub_idx] = Ci - m

    # --------------------------------------------
    # Compute Rw and Rb
    # --------------------------------------------
    Rw = np.zeros((n_ch, n_ch))
    Rb = np.zeros((n_ch, n_ch))

    for ep_idx in range(n_epochs):
        for i in range(n_trials):

            Ci = Epochs_covW[:, :, ep_idx, i]
            Rw += Ci @ Ci

            for j in range(i + 1, n_trials):
                Cj = Epochs_covW[:, :, ep_idx, j]
                Rb += Ci @ Cj

    # enforce symmetry
    Rw = (Rw + Rw.T) / 2
    Rb = (Rb + Rb.T) / 2

    # normalization
    Rw = Rw / (n_trials * n_epochs)
    Rb = Rb / (n_trials * (n_trials - 1) * n_epochs)

    # --------------------------------------------
    # Shrinkage
    # --------------------------------------------
    mean_eig = np.mean(np.linalg.eigvalsh(Rw))
    Rw = (1 - gamma) * Rw + gamma * mean_eig * np.eye(n_ch)

    # --------------------------------------------
    # Generalized eigenproblem
    # --------------------------------------------
    eigvals, w = eigh(Rb, Rw)

    # sort descending
    idx = np.argsort(eigvals)[::-1]
    eigvals = eigvals[idx]
    w = w[:, idx]

    # --------------------------------------------
    # Back-project to sensor space
    # --------------------------------------------
    Wpr = Wm @ w

    W = np.zeros_like(Wpr)
    A = np.zeros_like(Wpr)

    for comp_idx in range(Wpr.shape[1]):

        wi = Wpr[:, comp_idx]

        # normalize filter
        wi = wi / np.sqrt(wi.T @ Cxx @ wi)

        W[:, comp_idx] = wi

        # pattern recovery
        A[:, comp_idx] = (Cxx @ wi) / (wi.T @ Cxx @ wi)

    return W, A, eigvals

#####################################################################

def project_filters_to_manifold(V, Wm, Cxx):
    """
    Parameters
    ----------
    V : vectorized symmetric matrix (output of cov2upper)
    Wm : whitening matrix
    Cxx : mean covariance matrix

    Returns
    -------
    W : spatial filters
    A : spatial patterns
    S : eigenvalues
    """

    # --------------------------------------------
    # Recover symmetric matrix
    # --------------------------------------------
    WW = upper2cov(V)

    # --------------------------------------------
    # Eigen decomposition
    # --------------------------------------------
    S, Uw = eigh(WW)

    # sort descending
    idx = np.argsort(S)[::-1]
    S = S[idx]
    Uw = Uw[:, idx]

    # --------------------------------------------
    # Back-project + normalize
    # --------------------------------------------
    W = np.zeros_like(Uw)
    A = np.zeros_like(Uw)

    for local_src_idx in range(Uw.shape[1]):

        # Return filters from whitened space
        wi = Wm @ Uw[:, local_src_idx]

        # Normalize
        wi = wi / np.sqrt(wi.T @ Cxx @ wi)

        W[:, local_src_idx] = wi

        # Pattern recovery (forward model)
        A[:, local_src_idx] = (Cxx @ wi) / (wi.T @ Cxx @ wi)

    return W, A, S

#####################################################################
