close all
clear
clc

ft_path = 'C:\Users\anton\Documents\GitHub\CBI\site-packages\fieldtrip';

if ~exist('ft_defaults','file')
    addpath(ft_path);
end

ft_defaults;

%%
elec = load("D:\OS(CURRENT)\data\simulation_support_data\eeg\elec.mat").elec;

topo = [];
topo.dimord = 'chan_time';
topo.label  = elec.label;  
topo.time   = 0;
topo.elec   = elec;

laycfg = [];
laycfg.elec = elec;
lay = ft_prepare_layout(laycfg);     

cfg = [];
cfg.marker       = '';
cfg.layout       = lay;
cfg.comment      = 'no';
cfg.style        = 'fill';
cfg.markersymbol = '';
cfg.colorbar     = 'no'; 
cfg.layout.pos(:, 1:2) = cfg.layout.pos(:, 1:2) * 1.1; 
cfg.layout.pos(:, 2) = cfg.layout.pos(:, 2) - 0.05;

%%
G = load('D:\OS(CURRENT)\data\simulation_support_data\eeg\MNE_EEG_FWD_TRPL.mat').MNE_EEG_FWD_TRPL;

%%  
NConstSrc = 91; 
Ntg = 1; 
flanker = 1; 
TrLeSe = 5; 
Fs = 100; 
NTr = 50; 
NLclSrc = 11;
Wsize = 1 / 8; 
Ssize = Wsize/2;

%%
SNR = 3;

[Xtrials, Xraw, tm, TgPa] = gen_dat_corrca( ...
    G, NConstSrc, Ntg, flanker, TrLeSe, ...
    Fs, NTr, NLclSrc, SNR);

tmraw = repmat(tm,[1,NTr]);

%%
figure
plot(tm')

figure
topo.avg = TgPa;
ft_topoplotER(cfg, topo);

%%
mc_i = 1;

[W, A, Epochs_cov, z] = env_pca(Xtrials, Fs, Wsize, Ssize);

W = squeeze(W(1,:,:)); W = [W(:,1), W(:,end)];
A = squeeze(A(1,:,:)); A = [A(:,1), A(:,end)];

%%
env_corr = []; env_spoc = [];
for w_i = 1:2
    w = W(:,w_i);
    env_spoc(:,w_i) = abs(hilbert(w'*Xraw));
    env_corr(w_i) = corr(env_spoc(:,w_i),tmraw');
end
[b_corr_env, b_idx] = max(env_corr);
b_w = W(:,b_idx);
b_a = A(:,b_idx); 
b_env = env_spoc(:,b_idx);

env_corr_spoc(mc_i) = b_corr_env
patt_corr_spoc(mc_i) = abs(corr(b_a,TgPa))

%% FastICA
% [Aica, Wica] = fastica(Xraw, ...
%             'numOfIC', 60, ...
%             'verbose', 'off');
[Aica, Wica] = fastica(Xraw, ...
            'numOfIC', 30, ...
            'approach', 'symm', ... 
            'g', 'tanh', ...        
            'verbose', 'off');
Wica = Wica';

%%
env_corr = []; env_ica = [];
for w_i = 1:size(Wica, 2)
    w = Wica(:,w_i);
    env_ica(:,w_i) = abs(hilbert(w'*Xraw));
    env_corr(w_i) = corr(env_ica(:,w_i), tmraw');
end
[b_corr_env, b_idx] = max(env_corr);
b_w = Wica(:,b_idx);
b_a = Aica(:,b_idx);
b_env = env_ica(:,b_idx);

env_corr_ica(mc_i) = b_corr_env
patt_corr_ica(mc_i) = abs(corr(b_a,TgPa))

%%
figure
plot(mean(reshape(b_env,[],NTr),2))

figure
topo.avg = b_a;
ft_topoplotER(cfg, topo);

%%
