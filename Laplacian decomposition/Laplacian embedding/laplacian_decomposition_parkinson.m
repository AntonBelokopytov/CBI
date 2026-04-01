close all
clear
clc

ft_path = 'C:\Users\ansbel\Documents\2Git\fieldtrip';

if ~exist('ft_defaults','file')
    addpath(ft_path);
end

ft_defaults;

%% =====================================================================
% LOAD DATA
% =====================================================================
sub_path = 'sub2_center_out_epochs.fif';

cfg = [];
cfg.dataset = sub_path;
Xinf = ft_preprocessing(cfg);    % Load EEG/MEG data
Fs = Xinf.fsample;               % Sampling frequency

% Initialize topography structure
topo = [];
topo.dimord = 'chan_time';
topo.label  = Xinf.elec.label;  
topo.time   = 0;
topo.elec   = Xinf.elec;
topo.time    = 0;

% Prepare FieldTrip layout for topography plotting
laycfg = [];
laycfg.elec = Xinf.elec;
lay = ft_prepare_layout(laycfg);     

cfg.marker       = 'labels';
cfg.layout       = lay;
cfg.comment      = 'no';
cfg.style        = 'fill';
cfg.markersymbol = 'o';
cfg.colorbar     = 'yes'; 

%% =====================================================================
% BANDPASS FILTERING (ALPHA BAND)
% =====================================================================
[b,a] = butter(3,[30,45]/(Fs/2));   % 3rd-order Butterworth filter
n_channels = 38;                   % Only EEG channels

Xfilt = [];
Epfilt = [];
Epochs_filt = [];

for i = 1:numel(Xinf.trial)
    i
    Ep_raw = Xinf.trial{i}(1:n_channels,:);            
    Epfilt  = filtfilt(b,a,Ep_raw')';    % Zero-phase filtering
    Epfilt = Epfilt(:,Fs/2:end-Fs/2);    % Trim edges
    
    Xfilt = cat(2,Xfilt,Epfilt);         % Concatenate for SVD
    Epochs_filt(:,:,i) = Epfilt;         % Store filtered epoch
end

%% =====================================================================
% SVD AND PCA
% =====================================================================
[Upca,S,~] = svd(Xfilt,'econ');           % Singular Value Decomposition
S = diag(S);

% Estimate effective rank
tol = max(size(Xfilt)) * eps(S(1));
r = sum(S > tol);

% Cumulative variance explained
ve = S.^2;
var_explained = cumsum(ve) / sum(ve);
var_explained(end) = 1;

% Number of components explaining at least 99% variance
n_components = find(var_explained>=0.999, 1);
n_components = max(min(n_components, r), 1);

Upca = Upca(:,1:n_components);               % Keep relevant PCA components

% Project epochs onto PCA components
Epfilt_pca = [];
for i = 1:size(Epochs_filt,3)
    Epfilt_pca(:,:,i) = Upca'*Epochs_filt(:,:,i);
end

%% =====================================================================
% EPOCH SEGMENTATION
% =====================================================================

Wsize = 0.5;  % Window size (s)
Ssize = 0.5;  % Step size (s)

X_epo = []; 
time = [0];

for i = 1:size(Epfilt_pca,3)

    % Segment each trial into short overlapping windows
    ep_wins = epoch_data(Epfilt_pca(:,:,i)', Fs, Wsize, Ssize);

    % Concatenate windows across trials
    X_epo = cat(3, X_epo, ep_wins); 

    % Build time axis (center of each window)
    timeline = time(end) + (Wsize - Ssize) + (1:size(ep_wins,3))/2;
    time = [time, timeline];
end

time = time(2:end);   % remove initial zero

% =====================================================================
% Covariance matrices of segmented epochs
% =====================================================================

Covs = [];
for i = 1:size(X_epo,3)
    Covs(:,:,i) = cov(X_epo(:,:,i));   % covariance of each window
end

% Project covariance matrices to tangent space (Euclidean representation)
Tcovs = Tangent_space(Covs);

%%
% [emb,s] = laplace_embedding(Tcovs',10,10,3,'euclidean');

%%
Tcovs_real_imag = [real(Tcovs); imag(Tcovs)]; 

% Теперь pdist работает с обычными действительными числами, 
% но математически вычисляет точное комплексное расстояние!
Dists = squareform(pdist(Tcovs_real_imag', 'euclidean'));

% Dists = squareform(pdist(Tcovs', 'euclidean'));

%%
N_neigb = 20;
gamma = 10;

W = exp(-(Dists.^2) / (2 * gamma^2));
W = W - diag(diag(W));

W_n = zeros(size(Dists));
for i=1:size(Dists,1)
    [mvals, mids] = sort(W(i,:),'descend');
    W_n(i,mids(2:1+N_neigb)) = mvals(2:1+N_neigb);
end
W_n = (W_n + W_n') / 2;

D = diag(sum(W_n,2));
L = D - W_n;

%%
[U,S] = eigs(L, D, 1+10,'smallestreal');
S = diag(S);

U = U(:,2:end);

%%
scatter3(U(:,1),U(:,2),U(:,3))

%%
Cm = mean(Covs,3);
whitening_reg = 0.001;
Cm_r = Cm+whitening_reg*eye(size(Cm))*trace(Cm)/size(Cm,1);
iWm = sqrtm(Cm_r);    
Wm = eye(size(Cm_r,1)) / iWm;
% Wm = eye(size(Cm_r,1));

Covs_vec = [];
for i=1:size(X_epo,3)
    C = Covs(:,:,i);
    Covs_vec(i,:) = cov2upper(Wm*C*Wm');
end

%%
Covs_vec_mean = mean(Covs_vec, 1);
Covs_vec_centered = Covs_vec - Covs_vec_mean;

vLv = Covs_vec_centered' * L * Covs_vec_centered;
vDv = Covs_vec_centered' * D * Covs_vec_centered;

%%
whitening_reg = 0.00001;
vDv_reg = vDv+whitening_reg*eye(size(vDv))*trace(vDv)/size(vDv,1);
[V,S] = eig(vLv,vDv_reg,'chol');
S = diag(S); [S, idx] = sort(S,'ascend'); V = V(:,idx);

%%
threshold = 0;
valid_idx = S > threshold;

Af = cov(Covs_vec) * V(:,valid_idx);
% Af = V(:,valid_idx);
Wf = upper2cov(Af(:,1)); 
[Uw,Sw] = eig(Wf); [s,idx] = sort(diag(Sw),'descend');Uw=Uw(:,idx);

comp = 1;

w = Wm * Uw(:,comp);

Env = [];
for i = 1:size(Covs,3)
    Env(i) = w'*Covs(:,:,i)*w;
end

plot(Env)

A = Upca * Cm * Wm * Uw;

ax = A(:,comp);
topo.avg = ax;
ft_topoplotER(cfg, topo);

%%
Emb = Covs_vec_centered * V(:,valid_idx);
U = (U - mean(U, 1)) ./ std(U, 0, 1);
Emb = (Emb - mean(Emb, 1)) ./ std(Emb, 0, 1);

plot(Emb(:,1))

%%
scatter3(U(:,1),U(:,2),U(:,3))
hold on
scatter3(Emb(:,1),Emb(:,2),Emb(:,3))

%%
scatter3(Emb(:,1),Emb(:,2),Emb(:,3))

%% =====================================================================
% ИЗВЛЕЧЕНИЕ ENV ДЛЯ ПЕРВЫХ 3-Х КОМПОНЕНТ ГРАФА (ПОЛОЖИТЕЛЬНЫЕ И ОТРИЦАТЕЛЬНЫЕ)
% =====================================================================
num_graph_comps = 3;
Env_3d_pos = zeros(size(Covs, 3), num_graph_comps);
Env_3d_neg = zeros(size(Covs, 3), num_graph_comps);

for k = 1:num_graph_comps
    % Берем k-й паттерн из Haufe трансформации
    Wf_k = upper2cov(Af(:, k)); 
    
    % Ищем пространственные фильтры для этого паттерна
    [Uw_k, Sw_k] = eig(Wf_k); 
    [~, idx_k] = sort(diag(Sw_k), 'descend'); 
    Uw_k = Uw_k(:, idx_k);
    
    % Берем самый сильный "положительный" фильтр (первый)
    w_pos = Wm * Uw_k(:, 1); 
    
    % Берем самый сильный "отрицательный" фильтр (последний)
    w_neg = Wm * Uw_k(:, end);
    
    % Считаем огибающую (мощность) для каждой эпохи для обоих фильтров
    for i = 1:size(Covs, 3)
        Env_3d_pos(i, k) = w_pos' * Covs(:,:,i) * w_pos;
        Env_3d_neg(i, k) = w_neg' * Covs(:,:,i) * w_neg;
    end
end

% =====================================================================
% ВИЗУАЛИЗАЦИЯ 1: ВРЕМЕННАЯ ДИНАМИКА (1D Timeline)
% =====================================================================
% Сравниваем первую компоненту (k=1)
U_1_z       = (U(:, 1) - mean(U(:, 1))) / std(U(:, 1));
Emb_1_z     = (Emb(:, 1) - mean(Emb(:, 1))) / std(Emb(:, 1));
Env_1_pos_z = (Env_3d_pos(:, 1) - mean(Env_3d_pos(:, 1))) / std(Env_3d_pos(:, 1));
Env_1_neg_z = (Env_3d_neg(:, 1) - mean(Env_3d_neg(:, 1))) / std(Env_3d_neg(:, 1));

figure;
plot(U_1_z, 'b', 'LineWidth', 1.5, 'DisplayName', 'Non-linear Graph (U)');
hold on;
plot(Emb_1_z, 'g--', 'LineWidth', 1.5, 'DisplayName', 'Linear Vec-Cov (Emb)');
plot(Env_1_pos_z, 'r', 'LineWidth', 2, 'DisplayName', 'Positive Filter Power');
plot(Env_1_neg_z, 'm', 'LineWidth', 2, 'DisplayName', 'Negative Filter Power');
grid on;
legend('Location', 'best');
title('Динамика 1-й компоненты: Положительный vs Отрицательный полюс');
xlabel('Номер эпохи');
ylabel('Z-score (Амплитуда/Мощность)');

% =====================================================================
% ВИЗУАЛИЗАЦИЯ 2: 3D ПРОСТРАНСТВО (Procrustes)
% =====================================================================
% Стандартизируем все 3D пространства
U_3d_z       = zscore(U(:, 1:3));
Env_3d_pos_z = zscore(Env_3d_pos);
Env_3d_neg_z = zscore(Env_3d_neg);

% Подгоняем оба пространства огибающих под идеальный граф Лапласа
[d_pos, Z_pos] = procrustes(U_3d_z, Env_3d_pos_z);
[d_neg, Z_neg] = procrustes(U_3d_z, Env_3d_neg_z);

figure;
% 1. Оригинальный нелинейный граф (Идеал - Синий)
scatter3(U_3d_z(:,1), U_3d_z(:,2), U_3d_z(:,3), 50, 'b', 'filled', 'MarkerFaceAlpha', 0.4);
hold on;

% 2. Координаты от положительных фильтров (Красный)
scatter3(Z_pos(:,1), Z_pos(:,2), Z_pos(:,3), 40, 'r', 'filled', 'MarkerFaceAlpha', 0.8);

% 3. Координаты от отрицательных фильтров (Маджента)
scatter3(Z_neg(:,1), Z_neg(:,2), Z_neg(:,3), 40, 'm', 'filled', 'MarkerFaceAlpha', 0.8);

grid on;
view(3);
title(sprintf('3D Вложение: Граф vs Pos Filter (Err: %.3f) vs Neg Filter (Err: %.3f)', d_pos, d_neg));
legend('Graph (U)', 'Pos Filter (Env)', 'Neg Filter (Env)', 'Location', 'best');

%%
