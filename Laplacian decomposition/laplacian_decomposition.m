close all
clear
clc

ft_path = 'C:\Users\ansbel\Documents\2Git\fieldtrip';

if ~exist('ft_defaults','file')
    addpath(ft_path);
end

ft_defaults;

%%
sub_path = 'Tumyalis_music_epochs.fif';

cfg = [];
cfg.dataset = sub_path;
Xinf = ft_preprocessing(cfg);
Fs = Xinf.fsample;

topo = [];
topo.dimord = 'chan_time';
topo.label  = Xinf.elec.label;  
topo.time   = 0;
topo.elec   = Xinf.elec;

laycfg = [];
laycfg.elec = Xinf.elec;
lay = ft_prepare_layout(laycfg);     

cfg.marker       = 'labels';
cfg.layout       = lay;
cfg.comment      = 'no';
cfg.style        = 'fill';
cfg.markersymbol = 'o';
cfg.colorbar     = 'no'; 

%%
conditions = {'(1) RS EC 1', '(2) RS EO 1', '(3) 2Hz', '(4) 05Hz', '(5) 4Hz', '(6) 1Hz', '(7) 3Hz', ...
              '(8) NoRy 1','(9) Waltz 1','(10) Waltz 2','(11) NoRy 2','(12) NoRy 3','(13) Waltz 3', ...
              '(14) NoRy 4','(15) Waltz 4','(16) NoRy 5','(17) Waltz 5','(18) RS EC 2','(19) RS EO 2', ...
              '(20) Waltz 6','(21) Waltz 7','(22) Waltz 8'};

%% =====================================================================
% BANDPASS FILTERING
% =====================================================================
[b,a] = butter(3,[15,25]/(Fs/2));   % 3rd-order Butterworth filter
n_channels = 38;                    % Only EEG channels

Xfilt = [];
Epfilt = [];
Epochs_filt = [];

time_series = []; en_t = 0;
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
n_components = find(var_explained>=1, 1);
n_components = max(min(n_components, r), 1);

Upca = Upca(:,1:n_components);               % Keep relevant PCA components

% Project epochs onto PCA components
Epfilt_pca = [];
for i = 1:size(Epochs_filt,3)
    Epfilt_pca(:,:,i) = Upca'*Epochs_filt(:,:,i);
end

Xfiltpca = Upca'*Xfilt;

%% =====================================================================
% EPOCH SEGMENTATION
% =====================================================================
Wsize = 2;  % Window size in seconds
Ssize = 0.5;  % Step size in seconds

X_epo = []; X_epo_init = [];
time_series_epochs = [];
for i=1:size(Epfilt_pca,3)
    i
    ep_wins = epoch_data(Epfilt_pca(:,:,i)', Fs, Wsize, Ssize);
    X_epo = cat(3,X_epo,ep_wins); 

    ep_wins = epoch_data(Epochs_filt(:,:,i)', Fs, Wsize, Ssize);
    X_epo_init = cat(3,X_epo_init,ep_wins); 
end

Covs = []; Covs_vec = [];
for i=1:size(X_epo,3)
    C = cov(X_epo(:,:,i));
    Covs(:,:,i) = C;
    Covs_vec(i,:) = cov2upper(C);
end

Cm = mean(Covs,3);
whitening_reg = 0.001;
Cm_r = Cm+whitening_reg*eye(size(Cm))*trace(Cm)/size(Cm,1);
iWm = sqrtm(Cm_r);    
Wm = eye(size(Cm_r,1)) / iWm;

CovsW = []; Covs_vecW = [];
for i=1:size(X_epo,3)
    C = Covs(:,:,i);
    C = Wm*C*Wm';
    CovsW(:,:,i) = C;
    Covs_vecW(i,:) = cov2upper(C);
end

Tcovs = Tangent_space(CovsW);           

%%
n = size(Covs,3);
Dists = zeros(n,n);

for i = 1:n-1
    i
    Ai_inv_sqrt = inv(chol(Covs(:,:,i), 'lower'));

    parfor j = i+1:n
        B = Covs(:,:,j);

        X = Ai_inv_sqrt * B * Ai_inv_sqrt';
        e = eig(X);

        Dists(i,j) = sqrt(sum(log(max(e, eps)).^2));
    end
end

Dists = Dists + Dists.';

%%
% [emb,s] = laplace_embedding(Tcovs',10,10,3,'euclidean');

%%
DistsT = squareform(pdist(Tcovs', 'euclidean'));

%%
N_neigb = 10;
gamma = 10;

W = exp(-(DistsT.^2) / (2 * gamma^2));
W = W - diag(diag(W));

W_n = zeros(size(DistsT));
for i=1:size(DistsT,1)
    [mvals, mids] = sort(W(i,:),'descend');
    W_n(i,mids(2:1+N_neigb)) = mvals(2:1+N_neigb);
end
W_n = (W_n + W_n') / 2;

D = diag(sum(W_n,2));
L = D - W_n;

[U,S] = eigs(L, D, 1+10,'smallestreal');
S = diag(S);

U = U(:,2:end);

scatter3(U(:,1),U(:,2),U(:,3))

%%
[Uc,Sc,~] = svd(Covs_vecW','econ');
Uc = Uc(:,1:60);

Covs_vecW_pca = Covs_vecW * Uc;

%%
Covs_vec_mean = mean(Covs_vecW, 1);
Covs_vec_centered = Covs_vecW - Covs_vec_mean;

vLv = Covs_vec_centered' * L * Covs_vec_centered;
vDv = Covs_vec_centered' * D * Covs_vec_centered;

%%
whitening_reg = 0.;
vDv_reg = vDv+whitening_reg*eye(size(vDv))*trace(vDv)/size(vDv,1);
[V,S] = eig(vLv,vDv_reg,'chol');
S = diag(S); [S, idx] = sort(S,'ascend'); V = V(:,idx);

%%
threshold = 0;
valid_idx = S > threshold;

Af = cov(Covs_vecW) * V(:,valid_idx);
% Af = Uc * V(:,valid_idx);
Wf = upper2cov(Af(:,1)); 
[Uw,Sw] = eig(Wf); [s,idx] = sort(diag(Sw),'descend');Uw=Uw(:,idx);

comp = 1;

w = Wm * Uw(:,comp);

Env = [];
for i = 1:size(Covs,3)
    Env(i) = w' * Covs(:,:,i) * w;
end

Env = (Env - mean(Env)) / std(Env);
plot(Env)

A = Upca * Cm * Wm * Uw;

ax = A(:,comp);
topo.avg = ax;
ft_topoplotER(cfg, topo);

%%
Vv = V(:,valid_idx);
Emb = Covs_vec_centered * Vv;
U = (U - mean(U, 1)) ./ std(U, 0, 1);
Emb = (Emb - mean(Emb, 1)) ./ std(Emb, 0, 1);

comp = 7;
plot(Emb(:,comp))
hold on
plot(U(:,comp))

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
%% =====================================================================
% Canonical space visualization (cluster means)
% =====================================================================
x = U(:,1);
y = U(:,2);
z = U(:,3);
N_epoch_trial = 235
% x = Rmean(:,1);
% y = Rmean(:,2);
% z = Rmean(:,3);

figure; set(gcf,'Color','w');

num_clusters = 22;
cmap = jet(num_clusters);

ccx=[]; ccy=[]; ccz=[];
mask = 1:N_epoch_trial;

% --- считаем центры ---
for i = 1:num_clusters
    if mask(end) <= numel(x)
        sc_x = x(mask);
        sc_y = y(mask);
        sc_z = z(mask);
    else
        sc_x = x(mask(1):end);
        sc_y = y(mask(1):end);
        sc_z = z(mask(1):end);
    end
    mask = mask + N_epoch_trial;

    ccx = [ccx, mean(sc_x)];
    ccy = [ccy, mean(sc_y)];
    ccz = [ccz, mean(sc_z)];
end

plot3(ccx, ccy, ccz, 'k', 'LineWidth', 1);
hold on; grid on

% --- легенда будет по центрам ---
legend_handles = gobjects(num_clusters,1);

% --- рисуем кластеры и центры ---
mask = 1:N_epoch_trial;
for i = 1:num_clusters
    if mask(end) <= numel(x)
        sc_x = x(mask);
        sc_y = y(mask);
        sc_z = z(mask);
    else
        sc_x = x(mask(1):end);
        sc_y = y(mask(1):end);
        sc_z = z(mask(1):end);
    end
    mask = mask + N_epoch_trial;

    cx = mean(sc_x);
    cy = mean(sc_y);
    cz = mean(sc_z);

    % точки кластера (без легенды)
    scatter3(sc_x, sc_y, sc_z, 10, ...
        repmat(cmap(i,:), length(sc_x), 1), ...
        'filled', ...
        'MarkerFaceAlpha', 0.3);
    hold on

    % центр (для легенды)
    legend_handles(i) = scatter3(cx, cy, cz, 120, cmap(i,:), 'filled');

    % номер с фоном
    text(cx, cy, cz, num2str(i), ...
        'FontSize', 16, ...
        'FontWeight', 'bold', ...
        'Color', 'k', ... % цвет букв
        'BackgroundColor', [0.95 0.95 0.95], ... % цвет фона
        'Margin', 0.00001, ... % отступ вокруг текста
        'HorizontalAlignment', 'center', ...
        'VerticalAlignment', 'middle');
end

conditions = {'(1) RS EC 1', '(2) RS EO 1', '(3) 2Hz', '(4) 05Hz', '(5) 4Hz', '(6) 1Hz', '(7) 3Hz', ...
              '(8) NoRy 1','(9) Waltz 1','(10) Waltz 2','(11) NoRy 2','(12) NoRy 3','(13) Waltz 3', ...
              '(14) NoRy 4','(15) Waltz 4','(16) NoRy 5','(17) Waltz 5','(18) RS EC 2','(19) RS EO 2', ...
              '(20) Waltz 6','(21) Waltz 7','(22) Waltz 8'};

legend(legend_handles, conditions, 'Location', 'northeastoutside');

view(-45, 30);

% xlabel('UMAP component 1')
% ylabel('UMAP component 2')
% zlabel('UMAP component 3')
xlabel('Canonical axis 1')
ylabel('Canonical axis 2')
zlabel('Canonical axis 3')
