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

%%
BADS = load('BADS.mat').BADS.Tumyalis
BADS = fix(BADS * Fs)

%% =====================================================================
% BANDPASS FILTERING
% =====================================================================
[b,a] = butter(3,[15,25]/(Fs/2));   % 3rd-order Butterworth filter
n_channels = 38;                    % Only EEG channels

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

X_epo = []; time = [];
for i=1:size(Epfilt_pca,3)
    i

    ep_wins = epoch_data(Epochs_filt(:,:,i)', Fs, Wsize, Ssize);
    X_epo = cat(3,X_epo,ep_wins); 
    
end

% Covariance matrices of epochs
Covs = []; Covs_vec = [];
for i=1:size(X_epo,3)
    C = cov(X_epo(:,:,i));
    Covs(:,:,i) = C;
    Covs_vec(i,:) = cov2upper(C);
    % Covs_vec(i,:) = C(:);
end

Cm = cov(Xfilt');

%% =====================================================================
% EPOCH SEGMENTATION
% =====================================================================
Wsize = 2;  % Window size in seconds
Ssize = 0.5;  % Step size in seconds

X_epo = []; time = [];
for i=1:size(Epfilt_pca,3)
    i
    
    ep_wins = epoch_data(Epfilt_pca(:,:,i)', Fs, Wsize, Ssize);
    X_epo = cat(3,X_epo,ep_wins); 
    
end

% Covariance matrices of epochs
Covs_pca = []; Covs_vec_pca = [];
for i=1:size(X_epo,3)
    C = cov(X_epo(:,:,i));
    Covs_pca(:,:,i) = C;
    Covs_vec_pca(i,:) = cov2upper(C);
end

Cm_pca = riemann_mean(Covs_pca);
Tcovs = Tangent_space(Covs_pca,Cm_pca);           % Tangent space projection
N_epoch_trial = size(ep_wins,3);

% %%
% [U,S] = eigs(L, D, 1+10,'smallestreal');
% S = diag(S);
% stem(S)
% 
% U = U(:,2:end);
% 
% %%
% scatter3(U(:,1),U(:,2),U(:,3))

%%
[U,S] = eigs(L, D, 1+10,'smallestreal');
U = U(:,2:end);
scatter3(U(:,1),U(:,2),U(:,3))

%%
[L, D, W_n, W] = laplace_embedding(Tcovs',10,10);

%%
[Uc,Sc,~] = svd(Covs_vec_pca','econ');
Uc = Uc(:,1:100);
Covs_vec_pca_pca = Covs_vec_pca * Uc;

vLv = Covs_vec_pca_pca' * L * Covs_vec_pca_pca;
vDv = Covs_vec_pca_pca' * D * Covs_vec_pca_pca;

[U,S] = eigs(vLv, vDv, 1+10,'smallestreal');
S = diag(S);
stem(S)

U = U(:,2:end);

Sources = U' * Covs_vec_pca_pca'; Sources = Sources';

figure
scatter3(Sources(:,1),Sources(:,2),Sources(:,3))

distance_riemann
%%
j = 2;
WW = upper2cov(Uc * U(:,j));

[WW,S] = eig(WW); [S,idx] = sort(diag(S),'descend'); WW = WW(:,idx);
figure
stem(S)

w = WW(:,1);
% plot(w)
% hold on

Env = [];
for i=1:size(Covs_pca,3)
    Env(i) = w'*Covs_pca(:,:,i)*w;
end

Env = (Env - mean(Env)) / std(Env);
Envs(:,j) = Env;

figure
plot(Env)

a = Cm * Upca * w;
topo.avg   = a;
ft_topoplotER(cfg, topo);

%%
% Выбираем первую компоненту графа LPP
comp_idx = 1; 
WW_mat = upper2cov(U(:, comp_idx));

% Извлекаем все собственные векторы и значения
[W_eig, S_matrix] = eig(WW_mat); 
[S_vals, idx] = sort(diag(S_matrix), 'descend'); 
W_eig = W_eig(:, idx);

N_epochs = size(Covs_pca, 3);
N_filters = length(S_vals);

Env_sum = zeros(1, N_epochs); 

% Накапливаем взвешенную сумму мощностей от всех фильтров
for m = 1:30
    w_m = W_eig(:, m);
    lambda_m = S_vals(m);
    
    for i = 1:N_epochs
        Env_sum(i) = Env_sum(i) + lambda_m * (w_m' * Covs_pca(:,:,i) * w_m);
    end
end

% Z-score нормализация нашей суммы
Env_sum = (Env_sum - mean(Env_sum)) / std(Env_sum);

% Z-score нормализация оригинального Source (из векторного умножения)
% (Убедитесь, что матрица Sources у вас рассчитана корректно)
Source_original = Sources(:, comp_idx); 
Source_original = (Source_original - mean(Source_original)) / std(Source_original);

% Визуальное доказательство
figure;
plot(Source_original, 'k', 'LineWidth', 4, 'DisplayName', 'Оригинальный Source (U^T * Covs\_vec)');
hold on;
plot(Env_sum, 'r--', 'LineWidth', 2, 'DisplayName', 'Сумма мощностей (Сумма \lambda * w^T C w)');
title(sprintf('Эквивалентность для LPP компоненты %d', comp_idx));
xlabel('Индекс эпохи');
ylabel('Амплитуда (z-score)');
legend;
grid on;

%%
scatter3(Envs(:,1),Envs(:,2),Envs(:,3))

%% =====================================================================
% Canonical space visualization (cluster means)
% =====================================================================
R =  U(:,1:end);
x = Sources(:,1);
y = Sources(:,2);
z = Sources(:,3);

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
