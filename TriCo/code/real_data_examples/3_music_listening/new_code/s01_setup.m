% =====================================================================
% CONFIGURATION & PARAMETERS
% =====================================================================
% --- Paths & Subject ---
sub_name = 'ShoOle';
ft_path  = 'C:\Users\anton\Documents\GitHub\CBI\site-packages\fieldtrip';
sub_path = ['D:/OS(CURRENT)/scripts/2Git/TriCo/data/external/music_listening/', sub_name, '_music_epochs.fif'];

% Динамическая загрузка нужного поля из BADS.mat
BADS = load('D:\OS(CURRENT)\data\music\exp2\BADS.mat').BADS.(sub_name);

% --- Filtering & Epoching Params ---
freq_band  = [15, 25];      % Bandpass frequencies
n_channels = 38;            % Only EEG channels
Wsize      = 2;             % Window size in seconds
Ssize      = 0.5;           % Step size in seconds
nMC = 200; % Perm tests

% --- Experimental Conditions ---
conditions = {'(1) RS EC 1', '(2) RS EO 1', '(3) 2Hz', '(4) 05Hz', '(5) 4Hz', ...
              '(6) 1Hz', '(7) 3Hz', '(8) NoRy 1','(9) Waltz 1','(10) Waltz 2', ...
              '(11) NoRy 2','(12) NoRy 3','(13) Waltz 3', '(14) NoRy 4', ...
              '(15) Waltz 4','(16) NoRy 5','(17) Waltz 5','(18) RS EC 2', ...
              '(19) RS EO 2', '(20) Waltz 6','(21) Waltz 7','(22) Waltz 8'};
nEpochs = length(conditions);

% =====================================================================
% FIELDTRIP INIT & DATA PREPARATION
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