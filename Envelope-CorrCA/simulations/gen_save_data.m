close all
clear
clc

ft_path = 'C:\Users\ansbel\Documents\2Git\fieldtrip';

if ~exist('ft_defaults','file')
    addpath(ft_path);
end

ft_defaults;

%%
elec = load("elec.mat").elec;

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
G = load('MNE_EEG_FWD_TRPL.mat').MNE_EEG_FWD_TRPL;

%%
snr_range = [0.2, 0.5, 1, 2, 4, 8, 10];
mc = 500;
  
Nsrc = 91; 
Ntg = 1; 
flanker = 1; 
Ts = 5; 
Fs = 100; 
nEps = 50; 
nLclSrc = 11;
Wsize = 0.125; 
Ssize = Wsize / 2;

parfor k = 1:total_iters
    current_snr = snr_flat(k);
    curr_iter = iter_flat(k);
        
    [Xtrials, Xraw, m_true, targetA_true] = gen_dat_corrca(G, Nsrc, ...
        Ntg, flanker, Ts, Fs, ...
        nEps, nLclSrc, current_snr);
end

%%