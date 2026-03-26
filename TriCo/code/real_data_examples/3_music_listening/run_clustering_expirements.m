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
half1 = false(1,size(Tcovs,2));

m = 1:117;
for i=1:22
    half1(m) = true;
    m = m + 235;
end

half2 = ~half1;

%% =====================================================================
% UMAP EMBEDDING
% =====================================================================
clear u
u1 = UMAP("n_neighbors",20,"n_components",3,"min_dist",0);
u1.metric = 'euclidean';
u1.target_metric = 'euclidean';

u2 = UMAP("n_neighbors",20,"n_components",3,"min_dist",0);
u2.metric = 'euclidean';
u2.target_metric = 'euclidean';



% Low-dimensional embedding of epochs
u1.fit(Tcovs(:,half1)');
u2.fit(Tcovs(:,half2)');



R11 = u1.transform(Tcovs(:,half1)');
R12 = u1.transform(Tcovs(:,half2)');

R21 = u2.transform(Tcovs(:,half1)');
R22 = u2.transform(Tcovs(:,half2)');



R11 = R11 - mean(R11,1);
R12 = R12 - mean(R12,1);

R21 = R21 - mean(R21,1);
R22 = R22 - mean(R22,1);

%%
R11r = permute(reshape(R11,117,22,3),[2 1 3]);
R21r = permute(reshape(R21,117,22,3),[2 1 3]);

R12r = permute(reshape(R12,118,22,3),[2 1 3]);
R22r = permute(reshape(R22,118,22,3),[2 1 3]);

C11 = squeeze(mean(R11r,2));
C21 = squeeze(mean(R21r,2));

C12 = squeeze(mean(R12r,2));
C22 = squeeze(mean(R22r,2));

%%
scatter3(R11(:,1),R11(:,2),R11(:,3)); hold on
scatter3(C11(:,1),C11(:,2),C11(:,3),'filled')

%%
n = size(R11,1);
W21 = procrustes_rotation(C21, C11);
W12 = procrustes_rotation(C12, C11);
W22 = procrustes_rotation(C22, C11);

R21 = R21 * W21;
R12 = R12 * W12;
R22 = R22 * W22;

%%
sim12 = proc_dist(C11,C12);
sim13 = proc_dist(C11,C21);
sim14 = proc_dist(C11,C22);
sim23 = proc_dist(C12,C21);
sim24 = proc_dist(C12,C22);
sim34 = proc_dist(C21,C22);

D = zeros(4);
D(1,2)=sim12; D(1,3)=sim13; D(1,4)=sim14;
D(2,3)=sim23; D(2,4)=sim24;
D(3,4)=sim34;
D = D + D';

labels = {'R11','R12','R21','R22'};

Y = mdscale(D,2);

figure
scatter(Y(:,1),Y(:,2),150,'filled')
text(Y(:,1),Y(:,2),labels,'FontSize',12,'VerticalAlignment','bottom')

axis equal
grid on
title('Similarity of embeddings (MDS)')

%%

%% =====================================================================
% VISUALIZE UMAP (2x2)
% rows = fit
% cols = transform
% =====================================================================

figure
set(gcf,'Color','w')
tiledlayout(2,2,'TileSpacing','compact','Padding','compact')

titles = {
    'Fit half1 → Transform half1', ...
    'Fit half1 → Transform half2', ...
    'Fit half2 → Transform half1', ...
    'Fit half2 → Transform half2'};

R_all = {R11, R12, R21, R22};

for k = 1:4
    nexttile
    R = R_all{k};
    
    scatter3(R(:,1),R(:,2),R(:,3), 20, 'filled')
    axis equal
    grid on
    
    xlabel('UMAP 1')
    ylabel('UMAP 2')
    zlabel('UMAP 3')
    title(titles{k})
end

%% =====================================================================
% VISUALIZE UMAP (canonical coordinates, 2x2)
% rows = fit
% cols = transform
% =====================================================================

figure
set(gcf,'Color','w')
tiledlayout(2,2,'TileSpacing','compact','Padding','compact')

R_all = {R11, R12, R21, R22};
titles = {
    'Fit half1 → Transform half1', ...
    'Fit half1 → Transform half2', ...
    'Fit half2 → Transform half1', ...
    'Fit half2 → Transform half2'};

N_trials = [117 117 118 118];   % важно
num_clusters = 22;
cmap = jet(num_clusters);

for k = 1:4
    
    nexttile
    R = R_all{k};
    N_epoch_trial = N_trials(k);
    
    % if mod(k,2)==1
    %     X_current = X_epo(:,:,half1);
    % else
    %     X_current = X_epo(:,:,half2);
    % end
    % 
    % % ---- ESPOC ----
    % [W, A, Vf, Vz, corrs, VecCov, Epochs_cov, eigenvalues] = ...
    %     espoc(X_current, R');
    % 
    % % ---- Canonical coordinates ----
    % canonical = Vz(1:3,:) * R';

    x = R(:,1);
    y = R(:,2);
    z = R(:,3);

    hold on
    grid on

    ccx=[]; ccy=[]; ccz=[];
    legend_handles = gobjects(num_clusters,1);

    % ---- Compute & plot clusters ----
    for i = 1:num_clusters
        
        idx_start = (i-1)*N_epoch_trial + 1;
        idx_end   = min(i*N_epoch_trial, length(x));
        
        sc_x = x(idx_start:idx_end);
        sc_y = y(idx_start:idx_end);
        sc_z = z(idx_start:idx_end);

        % cluster center
        cx = mean(sc_x);
        cy = mean(sc_y);
        cz = mean(sc_z);

        ccx = [ccx cx];
        ccy = [ccy cy];
        ccz = [ccz cz];

        % cluster points
        scatter3(sc_x, sc_y, sc_z, 10, ...
            repmat(cmap(i,:), length(sc_x), 1), ...
            'filled', ...
            'MarkerFaceAlpha', 0.3);

        % center point
        legend_handles(i) = scatter3(cx, cy, cz, 120, cmap(i,:), 'filled');

        % label
        text(cx, cy, cz, num2str(i), ...
            'FontSize', 14, ...
            'FontWeight', 'bold', ...
            'Color', 'k', ...
            'BackgroundColor', [0.95 0.95 0.95], ...
            'HorizontalAlignment', 'center', ...
            'VerticalAlignment', 'middle');
    end

    % connect centers
    plot3(ccx, ccy, ccz, 'k', 'LineWidth', 1.2);

    title(titles{k})
    xlabel('UMAP axis 1')
    ylabel('UMAP axis 2')
    zlabel('UMAP axis 3')
    view(-45,30)
    axis tight
end

legend(legend_handles, conditions, 'Location','northeastoutside')

%% =====================================================================
% eSPoC COMPUTATION
% =====================================================================

% Run eSPoC
[W, A, Vf, Vz, corrs, VecCov, Epochs_cov, eigenvalues] = espoc(X_epo(:,:,half1), R21');

% Plot correlation values
figure;
set(gcf,'Color','w');

stem(corrs','LineWidth',1.5);
grid on

xlabel('Local component index')
ylabel('Correlation')
title('eSPoC correlation values')

legend({'Global 1','Global 2','Global 3'}, 'Location','best')

xlim([1 size(corrs,2)])

%% ====================================================================
% Canonical projections and cluster structure in canonical space
% =====================================================================
R = R21;
N_epoch_trial = 117;

% Global covariance activation (canonical components in feature space)
gl_src = Vf' * VecCov;

% Canonical projection of embedding
emb_can_pr = Vz' * R';

% Correlation between canonical covariance activations
% and canonical embedding projections
corr(gl_src', emb_can_pr')

figure; 
set(gcf,'Color','w');
tiledlayout(3,2)

% =====================================================================
% 3D visualization of canonical space (cluster means)
% =====================================================================

nexttile(1,[3,1])

% Canonical coordinates
x = Vz(:,1)' * R';
y = Vz(:,2)' * R';
z = Vz(:,3)' * R';

num_clusters = 22;
cmap = jet(num_clusters);

ccx=[]; ccy=[]; ccz=[];
mask = 1:N_epoch_trial;

% --- Compute cluster centers ---
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

% Connect cluster centers
plot3(ccx, ccy, ccz, 'k', 'LineWidth', 1);
hold on; grid on

legend_handles = gobjects(num_clusters,1);

% --- Plot clusters and their centers ---
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

    % Cluster points
    scatter3(sc_x, sc_y, sc_z, 10, ...
        repmat(cmap(i,:), length(sc_x), 1), ...
        'filled', ...
        'MarkerFaceAlpha', 0.3);

    % Cluster center (for legend)
    legend_handles(i) = scatter3(cx, cy, cz, 120, cmap(i,:), 'filled');

    % Label cluster index
    text(cx, cy, cz, num2str(i), ...
        'FontSize', 16, ...
        'FontWeight', 'bold', ...
        'Color', 'k', ...
        'BackgroundColor', [0.95 0.95 0.95], ...
        'HorizontalAlignment', 'center', ...
        'VerticalAlignment', 'middle');
end

legend(legend_handles, conditions, 'Location', 'northeastoutside');

% view(-45, 30);
xlabel('Canonical axis 1')
ylabel('Canonical axis 2')
zlabel('Canonical axis 3')

% =====================================================================
% Temporal dynamics of canonical components
% =====================================================================

tstep = N_epoch_trial;
ticks = 0:tstep:size(R,1);

for i = 1:3

    nexttile(i*2)

    % Normalize signals for visualization
    gl_src_n = (gl_src(i,:) - mean(gl_src(i,:))) / std(gl_src(i,:));
    emb_can_pr_n = (emb_can_pr(i,:) - mean(emb_can_pr(i,:))) / std(emb_can_pr(i,:));

    plot(gl_src_n,'blue')
    hold on
    plot(emb_can_pr_n,'red')

    title(['component ', num2str(i), ...
           ' | corr = ', ...
           num2str(corr(gl_src_n',emb_can_pr_n'),'%.2f')])

    grid()

    xticks(ticks(1:end-1));
    xlim([0, ticks(end)])

    % Condition indices on x-axis
    conditions_num = arrayfun(@(x) ['(' num2str(x) ')'], ...
                              1:22, ...
                              'UniformOutput', false);
    xticklabels(conditions_num);

    if i == 1
        legend('Global source signal', 'UMAP canonical projection')
    end

    if i == 3
        xlabel('Experimental conditions')
    end
end

% Replace numeric labels with condition names
xticklabels(conditions);

%% =====================================================================
% Canonical space visualization (cluster means)
% =====================================================================
R =  Rmean1;
x = Vz(:,1)'*R';
y = Vz(:,2)'*R';
z = Vz(:,3)'*R';

% x = R(:,1);
% y = R(:,2);
% z = R(:,3);

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

%% =====================================================================
% eSPoC COMPUTATION
% =====================================================================

% Run eSPoC
[W, A, Vf, Vz, corrs, VecCov, Epochs_cov, eigenvalues] = espoc(X_epo(:,:,half1), R21');

% Plot correlation values
figure;
set(gcf,'Color','w');

stem(corrs','LineWidth',1.5);
grid on

xlabel('Local component index')
ylabel('Correlation')
title('eSPoC correlation values')

legend({'Global 1','Global 2','Global 3'}, 'Location','best')

xlim([1 size(corrs,2)])

%%
R = R12;
N_epoch_trial = 117;

% Select global source and local component
gl_src_idx  = 1;
lcl_src_idx = 1;

% Back-project spatial pattern and spatial filter to sensor space
ax = U*A(gl_src_idx,:,lcl_src_idx)';
wx = U*W(gl_src_idx,:,lcl_src_idx)';

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Fix sign ambiguity (make dominant coefficient positive)
[~, idx] = max(abs(wx));
wx = wx.*sign(wx(idx));

[~, idx] = max(abs(ax));
ax = ax.*sign(ax(idx));

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

figure
set(gcf,'Color','w');

% Layout: filter | pattern | dynamics
t = tiledlayout(2,2,'TileSpacing','compact','Padding','compact');

% Display correlation value in title
sgtitle(['Source Envelope - UMAP correlation: ', ...
         num2str(corrs(gl_src_idx,lcl_src_idx))])

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% ---- Filter topography ----
ax1 = nexttile(t,1); 
title(ax1,'Filter');

topo.avg   = wx;
cfg.figure = ax1;
ft_topoplotER(cfg, topo); 

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% ---- Pattern topography ----
ax2 = nexttile(t,2);       
title(ax2,'Pattern');

topo.avg   = ax;
cfg.figure = ax2;
ft_topoplotER(cfg, topo); 

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% ---- Source dynamics vs canonical projection ----
ax3 = nexttile(t,3,[1,2]);
hold on; grid on        
title(ax3,'Latent Source Signal & UMAP canonical projection');

% Compute source power from covariance matrices
S = [];
for i = 1:size(Epochs_cov,3)
    S(i) = wx' * U * Epochs_cov(:,:,i) * U' * wx;
end

% Z-score normalization
S = (S-mean(S))/std(S);
plot(S,'LineWidth',1)

% Canonical projection of embedding
zz = Vz(:,gl_src_idx)'*R';
zz = (zz-mean(zz))/std(zz) * sign(corrs(gl_src_idx,lcl_src_idx));
plot(zz,'LineWidth',1,'Color','red')

% Condition boundaries on x-axis
tstep = N_epoch_trial;
ticks = 0:tstep:size(S,2);

xticks(ticks(1:end-1));
xlim([0 ticks(end)])

% Condition labels
conditions = {'(1) RS EC 1','(2) RS EO 1','(3) 2Hz','(4) 0.5Hz',...
              '(5) 4Hz','(6) 1Hz','(7) 3Hz','(8) NoRy 1','(9) Waltz 1',...
              '(10) Waltz 2','(11) NoRy 2','(12) NoRy 3','(13) Waltz 3',...
              '(14) NoRy 4','(15) Waltz 4','(16) NoRy 5','(17) Waltz 5',...
              '(18) RS EC 2','(19) RS EO 2','(20) Waltz 6',...
              '(21) Waltz 7','(22) Waltz 8'};

xticklabels(conditions);
xtickangle(45);

xlabel('Experimental conditions')
ylabel('Source signal')

legend('Source Signal Envelope','UMAP canonical projection')

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%
