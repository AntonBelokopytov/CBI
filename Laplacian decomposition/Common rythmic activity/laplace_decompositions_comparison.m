close all
clear
clc

ft_path = 'C:\Users\ansbel\Documents\GitHub\CBI\site-packages\fieldtrip';

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
NConstSrc = 40; 
Ntg = 1; 
flanker = 1; 
TrLeSe = 20; 
Fs = 100; 
NTr = 50; 
NLclSrc = 2;

SNR = 10;

[Xtrials, Xraw, tm, TgPa] = gen_dat_lap_dec( ...
    G, NConstSrc, Ntg, flanker, TrLeSe, ...
    Fs, NTr, NLclSrc, SNR);

tmraw = repmat(tm,[1,NTr]);

%%
figure
plot(tm')

%%
% figure
% topo.avg = TgPa(1,:);
% ft_topoplotER(cfg, topo);

%%
Wsize = 0.125;      
Ssize = Wsize / 2;    
n_comps = 3;    

[A, W, z, Epochs_cov] = env_laplace_dec(Xtrials, Fs, Wsize, Ssize, [], [], n_comps);

%%
Env = [];
for tr_i=1:size(Epochs_cov,4)
    w = squeeze(W(1,1,:,tr_i));
    Env(:,tr_i) = abs(hilbert(Xtrials(:,:,tr_i)*w));
end

%%
size(Env)
figure
imagesc(Env')

figure
plot(mean(Env,2))

%%
