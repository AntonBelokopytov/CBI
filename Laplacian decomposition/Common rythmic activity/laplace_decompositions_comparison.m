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
Ntg = 1; 
flanker = 1; 
TrLeSe = 10; 
Fs = 250; 
NTr = 100; 
NConstSrc = 90;
NLclSrc = 5;

SNR = 5;

% [Xtrials, Xraw, tm, TgPa] = gen_multisub( ...
%     G, Ntg, flanker, TrLeSe, ...
%     Fs, NTr, NLclSrc, SNR);

[Xtrials, Xraw, tm, TgPa] = gen_trials( ...
    G, NConstSrc, Ntg, flanker, TrLeSe, ...
    Fs, NTr, NLclSrc, SNR);

%%
topo.avg    = TgPa(1,:);
ft_topoplotER(cfg, topo);

%%
Wsize = 1;      
Ssize = Wsize / 5;    

tm_epochs = epoch_data(tm',Fs,Wsize, Ssize);
tm_epochs = squeeze(mean(tm_epochs,1));
tm_epochs = (tm_epochs - mean(tm_epochs)) / std(tm_epochs);

figure
plot(tm_epochs)

%%
z = env_laplace_dec(Xtrials, Fs, Wsize, Ssize);

%%
k = 1;

c = corr(z(:,k), tm_epochs)

figure
plot(sign(c) * z(:,k)')
hold on
plot(tm_epochs)

%%
