%% =====================================================================
% 1. CONFIGURATION & PARAMETERS
% =====================================================================
close all; clear; clc;

% --- Paths & Subject ---
sub_name = 'Tumyalis';
ft_path  = 'C:\Users\anton\Documents\GitHub\CBI\site-packages\fieldtrip';
sub_path = ['D:/OS(CURRENT)/scripts/2Git/TriCo/data/external/music_listening/', sub_name, '_music_epochs.fif'];

% Динамическая загрузка нужного поля из BADS.mat
BADS = load('D:\OS(CURRENT)\data\music\exp2\BADS.mat').BADS.(sub_name);

% --- Filtering & Epoching Params ---
freq_band  = [15, 25];       % Bandpass frequencies
n_channels = 38;            % Only EEG channels
Wsize      = 2;             % Window size in seconds
Ssize      = 0.5;           % Step size in seconds

% --- Experimental Conditions ---
conditions = {'(1) RS EC 1', '(2) RS EO 1', '(3) 2Hz', '(4) 05Hz', '(5) 4Hz', ...
              '(6) 1Hz', '(7) 3Hz', '(8) NoRy 1','(9) Waltz 1','(10) Waltz 2', ...
              '(11) NoRy 2','(12) NoRy 3','(13) Waltz 3', '(14) NoRy 4', ...
              '(15) Waltz 4','(16) NoRy 5','(17) Waltz 5','(18) RS EC 2', ...
              '(19) RS EO 2', '(20) Waltz 6','(21) Waltz 7','(22) Waltz 8'};
nEpochs = length(conditions);

%% =====================================================================
% 2. FIELDTRIP INIT & DATA PREPARATION
% =====================================================================
if ~exist('ft_defaults','file')
    addpath(ft_path);
end
ft_defaults;
cfg = [];
cfg.dataset = sub_path;
Xinf = ft_preprocessing(cfg);
Fs = Xinf.fsample;

laycfg = [];
laycfg.elec = Xinf.elec;
lay = ft_prepare_layout(laycfg);     

topo = [];
topo.dimord = 'chan_time';
topo.label  = Xinf.elec.label;  
topo.time   = 0;
topo.elec   = Xinf.elec;

cfg_topo = [];
cfg_topo.marker       = 'labels';
cfg_topo.layout       = lay;
cfg_topo.comment      = 'no';
cfg_topo.style        = 'fill';
cfg_topo.markersymbol = 'o';
cfg_topo.colorbar     = 'no'; 

%% =====================================================================
% 3. BANDPASS FILTERING & CONTINUOUS MASKING
% =====================================================================
[b,a] = butter(3, freq_band/(Fs/2));   
Xfilt = [];
Epochs_filt = [];
time_series = []; 
en_t = 0;

for i = 1:numel(Xinf.trial)
    Ep_raw = Xinf.trial{i}(1:n_channels,:);            
    Epfilt = filtfilt(b,a,Ep_raw')';         
    Epfilt = Epfilt(:,Fs/2:end-Fs/2);        
    ep_time = 1:size(Ep_raw,2); 
    
    time_series(:,i) = en_t + ep_time; 
    en_t = en_t + ep_time(end);
    
    Xfilt = cat(2,Xfilt,Epfilt);             
    Epochs_filt(:,:,i) = Epfilt;             
end
time_series = time_series(Fs/2:end-Fs/2,:);
time_series_raw = reshape(time_series,1,[]);

mask_ts = true(1,size(time_series_raw,2));
for i=1:size(BADS,1)
    bad_st = BADS(i,1);
    bad_en = bad_st + BADS(i,2);
    bad_idx = (time_series_raw >= bad_st) & (time_series_raw <= bad_en);
    mask_ts(bad_idx) = false;
end

%% =====================================================================
% 4. SVD AND PCA DIMENSIONALITY REDUCTION
% =====================================================================
[U,S,~] = svd(Xfilt(:,mask_ts),'econ');
S = diag(S);
tol = max(size(Xfilt)) * eps(S(1));
r = sum(S > tol);
ve = S.^2;
var_explained = cumsum(ve) / sum(ve);
var_explained(end) = 1;
n_components = find(var_explained>=1, 1);
n_components = max(min(n_components, r), 1);
U = U(:,1:n_components);               

Epfilt_pca = zeros(n_components, size(Epochs_filt,2), size(Epochs_filt,3));
for i = 1:size(Epochs_filt,3)
    Epfilt_pca(:,:,i) = U'*Epochs_filt(:,:,i);
end
Xfiltpca = U'*Xfilt;

%% =====================================================================
% 5. EPOCH SEGMENTATION & ARTIFACT MASKING
% =====================================================================
X_epo = []; time = [];
time_series_epochs = [];
cond_idx_epochs = []; 

for i=1:size(Epfilt_pca,3)
    ep_wins = epoch_data(Epfilt_pca(:,:,i)', Fs, Wsize, Ssize);
    X_epo = cat(3, X_epo, ep_wins); 
    
    ts_wins = epoch_data(time_series(:,i), Fs, Wsize, Ssize);
    time_series_epochs = cat(2, time_series_epochs, ts_wins); 
    
    cond_idx_epochs = [cond_idx_epochs, repmat(i, 1, size(ep_wins, 3))];
    
    timeline = 0.5 + ( Wsize/2:Ssize:(size(ep_wins,3)*Ssize+Ssize) );
    if i>1
        timeline = timeline + time(end) + Wsize-Ssize;
    end
    time = [time, timeline];
end

Covs = zeros(size(X_epo,2), size(X_epo,2), size(X_epo,3)); 
for i=1:size(X_epo,3)
    Covs(:,:,i) = cov(X_epo(:,:,i));
end
Tcovs = Tangent_space(Covs);           
N_epoch_trial = size(ep_wins,3);

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

valid_cond_idx = cond_idx_epochs(ep_mask);
boundaries = find(diff(valid_cond_idx) > 0);
ticks = [0, boundaries, length(valid_cond_idx)];
valid_Covs = Covs(:,:,ep_mask); % Только валидные матрицы для мощности

%% =====================================================================
% 6. UMAP EMBEDDING
% =====================================================================
u = UMAP("n_neighbors", 20, "n_components", 3, "min_dist", 0);
u.metric = 'euclidean';
u.target_metric = 'euclidean';
R = u.fit_transform(Tcovs(:,ep_mask)');
Rmean = R - mean(R,1);

%% =====================================================================
% FIGURE 1: VISUALIZE UMAP 3D SCATTER
% =====================================================================
figure; set(gcf, 'Color', 'w');
scatter3(R(:,1),R(:,2),R(:,3));
xlabel('UMAP component 1'); ylabel('UMAP component 2'); zlabel('UMAP component 3');
title('UMAP Embedding');

%% =====================================================================
% FIGURE 2: Temporal evolution of UMAP components
% =====================================================================
figure; set(gcf, 'Color', 'w');
plot(R)                          
xticks(ticks(1:end-1));
xlim([0, size(R,1)])
ylabel('UMAP component coordinate'); xlabel('Experimental conditions (Transitions)');
legend({'component 1', 'component 2', 'component 3'})
title('Temporal evolution of UMAP components');

%% =====================================================================
% eSPoC COMPUTATION
% =====================================================================
[W, A, Vf, Vz, corrs, VecCov, Epochs_cov, eigenvalues] = espoc(X_epo(:,:,ep_mask), R');

%% =====================================================================
% FIGURE 3: Plot correlation values
% =====================================================================
figure; set(gcf,'Color','w');
stem(corrs','LineWidth',1.5);
grid on
xlabel('Local component index'); ylabel('Correlation');
title('eSPoC correlation values');
legend({'Global 1','Global 2','Global 3'}, 'Location','best');
xlim([1 size(corrs,2)]);

%% =====================================================================
% FIGURE 4: Canonical projections and cluster structure (Tiled Layout)
% =====================================================================
gl_src = Vf' * VecCov;
emb_can_pr = Vz' * R';

figure; set(gcf,'Color','w');
tiledlayout(3,2)

% --- Left side: 3D cluster ---
nexttile(1,[3,1])
x = Vz(:,1)' * Rmean'; y = Vz(:,2)' * Rmean'; z = Vz(:,3)' * Rmean';
cmap = jet(nEpochs);
ccx=[]; ccy=[]; ccz=[]; mask = 1:N_epoch_trial;

for i = 1:nEpochs
    if mask(end) <= numel(x)
        sc_x = x(mask); sc_y = y(mask); sc_z = z(mask);
    else
        sc_x = x(mask(1):end); sc_y = y(mask(1):end); sc_z = z(mask(1):end);
    end
    mask = mask + N_epoch_trial;
    ccx = [ccx, mean(sc_x)]; ccy = [ccy, mean(sc_y)]; ccz = [ccz, mean(sc_z)];
end

plot3(ccx, ccy, ccz, 'k', 'LineWidth', 1); hold on; grid on
legend_handles = gobjects(nEpochs,1);
mask = 1:N_epoch_trial;

for i = 1:nEpochs
    if mask(end) <= numel(x)
        sc_x = x(mask); sc_y = y(mask); sc_z = z(mask);
    else
        sc_x = x(mask(1):end); sc_y = y(mask(1):end); sc_z = z(mask(1):end);
    end
    mask = mask + N_epoch_trial;
    cx = mean(sc_x); cy = mean(sc_y); cz = mean(sc_z);
    
    scatter3(sc_x, sc_y, sc_z, 10, repmat(cmap(i,:), length(sc_x), 1), 'filled', 'MarkerFaceAlpha', 0.3);
    legend_handles(i) = scatter3(cx, cy, cz, 120, cmap(i,:), 'filled');
    
    text(cx, cy, cz, num2str(i), 'FontSize', 16, 'FontWeight', 'bold', 'Color', 'k', ...
        'BackgroundColor', [0.95 0.95 0.95], 'HorizontalAlignment', 'center', 'VerticalAlignment', 'middle');
end
legend(legend_handles, conditions, 'Location', 'northeastoutside');
view(-45, 30);
xlabel('Canonical axis 1'); ylabel('Canonical axis 2'); zlabel('Canonical axis 3');

% --- Right side: Temporal dynamics ---
for i = 1:3
    nexttile(i*2)
    gl_src_n = (gl_src(i,:) - mean(gl_src(i,:))) / std(gl_src(i,:));
    emb_can_pr_n = (emb_can_pr(i,:) - mean(emb_can_pr(i,:))) / std(emb_can_pr(i,:));
    
    plot(gl_src_n,'blue'); hold on; plot(emb_can_pr_n,'red');
    title(['component ', num2str(i), ' | corr = ', num2str(corr(gl_src_n',emb_can_pr_n'),'%.2f')])
    grid on
    xticks(ticks(1:end-1)); xlim([0, ticks(end)])
    
    conditions_num = arrayfun(@(val) ['(' num2str(val) ')'], 1:nEpochs, 'UniformOutput', false);
    xticklabels(conditions_num);
    if i == 1, legend('Global source signal', 'UMAP canonical projection'); end
    if i == 3, xlabel('Experimental conditions'); end
end
xticklabels(conditions);

%% =====================================================================
% PERMUTATION TEST (CIRCULAR TIME SHIFTS)
% =====================================================================
chs = size(Xfiltpca, 1);
samples_per_epoch = size(Xfiltpca, 2) / nEpochs; 

nMC = 1;
corrmax = zeros(3, nMC);
corrmin = zeros(3, nMC);
disp('Running Permutation Test...');

parfor i = 1:nMC
    r_idx = fix(rand * size(Xfiltpca, 2));
    XCirc = circshift(Xfiltpca, [0, r_idx]);
    mask_ts_shifted = circshift(mask_ts, [0, r_idx]);
    
    Eps_circ = reshape(XCirc, chs, samples_per_epoch, nEpochs);
    mask_ts_shifted_eps = reshape(mask_ts_shifted, samples_per_epoch, nEpochs);
    
    X_test = [];
    for j = 1:nEpochs
        ep_wins = epoch_data(Eps_circ(:,:,j)', Fs, Wsize, Ssize);
        mask_ep_wins = epoch_data(double(mask_ts_shifted_eps(:,j)), Fs, Wsize, Ssize);
        valid_windows = all(mask_ep_wins, 1);
        X_test = cat(3, X_test, ep_wins(:,:,valid_windows));
    end
    
    neps = min(size(X_test,3),size(R,1));
    [~,~,~,~,corrs_perm] = espoc(X_test(:,:,1:neps), R(1:neps,:)'); 
    corrmax(:,i) = max(corrs_perm,[],2);
    corrmin(:,i) = min(corrs_perm,[],2);
end

corrmax1 = sort(max(corrmax,[],1),'descend');
corrmin1 = sort(min(corrmin,[],1));
alpha = 0.05;
i=1; while 1 - sum(corrmax1(i) > corrmax1)/numel(corrmax1) <= alpha, i=i+1; end; max_val = corrmax1(i);
i=1; while 1 - sum(corrmin1(i) < corrmin1)/numel(corrmin1) <= alpha, i=i+1; end; min_val = corrmin1(i);

%% =====================================================================
% FIGURE 5: SIGNIFICANCE THRESHOLDS
% =====================================================================
figure; stem(corrs'); hold on
yline(max_val, 'r', 'Alpha 0.05 Max'); yline(min_val, 'b', 'Alpha 0.05 Min');
title('Significance Thresholds');

%% =====================================================================
% FIGURE 6: Standalone Canonical space visualization
% =====================================================================
figure; set(gcf,'Color','w');
plot3(ccx, ccy, ccz, 'k', 'LineWidth', 1); hold on; grid on

legend_handles2 = gobjects(nEpochs,1);
mask = 1:N_epoch_trial;
for i = 1:nEpochs
    if mask(end) <= numel(x)
        sc_x = x(mask); sc_y = y(mask); sc_z = z(mask);
    else
        sc_x = x(mask(1):end); sc_y = y(mask(1):end); sc_z = z(mask(1):end);
    end
    mask = mask + N_epoch_trial;
    cx = mean(sc_x); cy = mean(sc_y); cz = mean(sc_z);
    
    scatter3(sc_x, sc_y, sc_z, 10, repmat(cmap(i,:), length(sc_x), 1), 'filled', 'MarkerFaceAlpha', 0.3); hold on
    legend_handles2(i) = scatter3(cx, cy, cz, 120, cmap(i,:), 'filled');
    text(cx, cy, cz, num2str(i), 'FontSize', 16, 'FontWeight', 'bold', 'Color', 'k', ...
        'BackgroundColor', [0.95 0.95 0.95], 'Margin', 0.00001, 'HorizontalAlignment', 'center', 'VerticalAlignment', 'middle');
end
legend(legend_handles2, conditions, 'Location', 'northeastoutside'); view(-45, 30);
xlabel('Canonical axis 1'); ylabel('Canonical axis 2'); zlabel('Canonical axis 3');

%% =====================================================================
% FIGURE 7: 2x2 Layout (Single Global & Local component)
% =====================================================================
gl_src_idx  = 3;
lcl_src_idx = 30;
ax = U*A(gl_src_idx,:,lcl_src_idx)';
wx = U*W(gl_src_idx,:,lcl_src_idx)';

[~, idx] = max(abs(wx)); wx = wx.*sign(wx(idx));
[~, idx] = max(abs(ax)); ax = ax.*sign(ax(idx));

figure; set(gcf,'Color','w');
t = tiledlayout(2,2,'TileSpacing','compact','Padding','compact');
sgtitle(['Source Envelope - UMAP correlation: ', num2str(corrs(gl_src_idx,lcl_src_idx))])

ax1 = nexttile(t,1); title(ax1,'Filter');
topo.avg = wx; cfg_topo.figure = ax1; ft_topoplotER(cfg_topo, topo); 

ax2 = nexttile(t,2); title(ax2,'Pattern');
topo.avg = ax; cfg_topo.figure = ax2; ft_topoplotER(cfg_topo, topo); 

ax3 = nexttile(t,3,[1,2]); hold on; grid on        
title(ax3,'Latent Source Signal & UMAP canonical projection');

S = zeros(1, size(valid_Covs,3));
for i = 1:size(valid_Covs,3)
    S(i) = wx' * U * valid_Covs(:,:,i) * U' * wx;
end
S = (S-mean(S))/std(S);

zz = Vz(:,gl_src_idx)'*Rmean';
zz = (zz-mean(zz))/std(zz) * sign(corrs(gl_src_idx,lcl_src_idx));

plot(S,'LineWidth',1); plot(zz,'LineWidth',1,'Color','red')
xticks(ticks(1:end-1)); xlim([0 ticks(end)])
xticklabels(conditions); xtickangle(45);
xlabel('Experimental conditions'); ylabel('Source signal')
legend('Source Signal Envelope','UMAP canonical projection')

%% =====================================================================
% FIGURE 8: 4x5 Layout (Multiple Components)
% =====================================================================
figure; set(gcf, 'Color', 'w');
t2 = tiledlayout(4,5,'TileSpacing','compact','Padding','compact');

comp_indices = [2 3 4 5];  
all_right_axes = [];       

for i = 1:length(comp_indices)
    lcl_src_idx = comp_indices(i);   
    ax = U*A(gl_src_idx,:,lcl_src_idx)';
    wx = U*W(gl_src_idx,:,lcl_src_idx)';
    
    [~, idx] = max(abs(wx)); wx = wx .* sign(wx(idx));
    [~, idx] = max(abs(ax)); ax = ax .* sign(ax(idx));
    row = (i-1)*5;
    
    ax_wx = nexttile(t2, row + 1);
    topo.avg = wx; cfg_topo.figure = ax_wx; ft_topoplotER(cfg_topo, topo);
    if i==1, title('Filter'); end
    
    ax_ax = nexttile(t2, row + 2);
    topo.avg = ax; cfg_topo.figure = ax_ax; ft_topoplotER(cfg_topo, topo);
    if i==1, title('Pattern'); end
    
    ax_plot = nexttile(t2, row + 3, [1 3]);
    all_right_axes = [all_right_axes ax_plot];
    
    S = zeros(1, size(valid_Covs,3));
    for j = 1:size(valid_Covs,3)
        S(j) = wx' * U * valid_Covs(:,:,j) * U' * wx;
    end
    S = (S - mean(S)) / std(S);
    
    plot(ax_plot, S, 'LineWidth', 1); hold on;
    plot(ax_plot, zz, 'LineWidth', 1, 'Color', 'red');
    grid on
    
    xticks(ax_plot, ticks(1:end-1)); xticklabels(ax_plot, []); xlim(ax_plot, [0, ticks(end)])
    title(ax_plot, ['corr = ', num2str(corrs(gl_src_idx,lcl_src_idx),'%.2f')])
end

conditions_num = arrayfun(@(val) ['(' num2str(val) ')'], 1:nEpochs, 'UniformOutput', false);

for i = 1:3
    ax_dyn = all_right_axes(i);
    xticklabels(ax_dyn, conditions_num);
end

ax_last = all_right_axes(end);
xticklabels(ax_last, conditions); xtickangle(ax_last, 45);
xlabel(ax_last, 'Experimental conditions')

for i = 1:4
    ax_dyn = all_right_axes(i);
    for k = 1:length(ticks(2:end-1))
        xline(ax_dyn, ticks(k+1), '--', 'Color', [0.1 0.1 0.1]);
    end
end
legend(all_right_axes(1), 'Source Signal Envelope', 'UMAP canonical projection');