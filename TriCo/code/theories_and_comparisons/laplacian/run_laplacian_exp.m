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

time_series = []; en_t = 0;
for i = 1:numel(Xinf.trial)
    i
    Ep_raw = Xinf.trial{i}(1:n_channels,:);            
    Epfilt  = filtfilt(b,a,Ep_raw')';    % Zero-phase filtering
    Epfilt = Epfilt(:,Fs/2:end-Fs/2);    % Trim edges

    ep_time = 1:size(Ep_raw,2); 
    time_series(:,i) = en_t + ep_time; en_t = en_t + ep_time(end);

    Xfilt = cat(2,Xfilt,Epfilt);         % Concatenate for SVD
    Epochs_filt(:,:,i) = Epfilt;         % Store filtered epoch
end
time_series = time_series(Fs/2:end-Fs/2,:);
time_series_raw = reshape(time_series,1,[]);

%%
mask_ts = true(1,size(time_series_raw,2));
for i=1:size(BADS,1)
    bad_st = BADS(i,1);
    bad_en = bad_st + BADS(i,2);

    bad_idx = (time_series_raw >= bad_st) & (time_series_raw <= bad_en);
    mask_ts(bad_idx) = false;
end

%% =====================================================================
% SVD AND PCA
% =====================================================================
[U,S,~] = svd(Xfilt(:,mask_ts),'econ');           % Singular Value Decomposition
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

U = U(:,1:n_components);               % Keep relevant PCA components

% Project epochs onto PCA components
Epfilt_pca = [];
for i = 1:size(Epochs_filt,3)
    Epfilt_pca(:,:,i) = U'*Epochs_filt(:,:,i);
end

Xfiltpca = U'*Xfilt;

%% =====================================================================
% EPOCH SEGMENTATION
% =====================================================================
Wsize = 2;  % Window size in seconds
Ssize = 0.5;  % Step size in seconds

X_epo = []; time = [];
time_series_epochs = [];
for i=1:size(Epfilt_pca,3)
    i

    ep_wins = epoch_data(Epfilt_pca(:,:,i)', Fs, Wsize, Ssize);
    X_epo = cat(3,X_epo,ep_wins); 
    
    ts_wins = epoch_data(time_series(:,i), Fs, Wsize, Ssize);
    time_series_epochs = cat(2,time_series_epochs,ts_wins); 

    timeline = 0.5 + ( Wsize/2:Ssize:(size(ep_wins,3)*Ssize+Ssize) );
    if i>1
        timeline = timeline + time(end) + Wsize-Ssize;
    end
    time = [time,timeline];
end

% Covariance matrices of epochs
Covs = []; Covs_vec = [];
for i=1:size(X_epo,3)
    C = cov(X_epo(:,:,i));
    Covs(:,:,i) = C;
    Covs_vec(i,:) = cov2upper(C);
    % Covs_vec(i,:) = C(:);
end

Tcovs = Tangent_space(Covs);           % Tangent space projection
N_epoch_trial = size(ep_wins,3);

%%
BADS;
ep_mask = true(1,size(X_epo,3));

for ep_idx=1:size(time_series_epochs,2)
    ep_st = time_series_epochs(1,ep_idx);
    ep_en = time_series_epochs(end,ep_idx);

    for i=1:size(BADS,1)
        bad_st = BADS(i,1);
        bad_en = BADS(i,1) + BADS(i,2);

        if (bad_st <= ep_en) && (bad_en >= ep_st)
            ep_mask(ep_idx) = false;
        end
    end
end

%%
% Wt = cov(Tcovs')^-0.5;

%%
[emb,s] = laplace_embedding(Tcovs',10,10,3,'eucledian');

%%
[emb,s] = laplace_embedding(Covs_vec,10,10,10,metric);

%%
Dists = zeros(size(Covs,3),size(Covs,3));
for i=1:size(Dists,1)-1
    i
    parfor j=i+1:size(Dists,2)
        d = distance_riemann(Covs(:,:,i),Covs(:,:,j));
        Dists(i,j) = d;
    end
end
Dists = (Dists + Dists') / 2;

%%
W = exp(-(Dists.^2) / (2 * 10^2));
W = W - diag(diag(W));

W_n = zeros(size(Dists));
for i=1:size(Dists,1)
    [mvals, mids] = sort(W(i,:),'descend');
    W_n(i,mids(2:1+N_neigb)) = mvals(2:1+N_neigb);
end
W_n = (W_n + W_n') / 2;

D = diag(mean(W_n,2));
L = D - W_n;

[U,S] = eigs(L, D, 1+10,'smallestreal');
S = diag(S);

U = U(:,2:end);


%%
% [emb,s] = laplace_embedding(Covs_vec,5,10);

%%
scatter3(emb(:,1),emb(:,2),emb(:,3))

%%
[U,S,~] = svd(Tcovs,'econ');
Tcovsdim = U'*Tcovs;

%%
[emb,s] = laplace_embedding(Tcovsdim',10,10,3);

%% =====================================================================
% Canonical space visualization (cluster means)
% =====================================================================
R =  U(:,1:end);
x = R(:,1);
y = R(:,2);
z = R(:,3);

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
