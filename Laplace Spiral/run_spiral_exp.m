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
phi1 = 0;
phi2 = pi;
f = 1;
Nc = 1.5; 
noise_level = 0.1; 
Ts = 20;
Fs = 250;
N = Ts*Fs;

A = linspace(0, 2*pi*f*Nc, N); 

x1 = A .* cos(A + phi1) + noise_level * randn(size(A));
y1 = A .* sin(A + phi1) + noise_level * randn(size(A));

x2 = A .* cos(A + phi2) + noise_level * randn(size(A));
y2 = A .* sin(A + phi2) + noise_level * randn(size(A));

min_val = min([x1, x2, y1, y2]);
shift_val = abs(min_val) + eps;

x1 = x1 + shift_val;
y1 = y1 + shift_val;
x2 = x2 + shift_val;
y2 = y2 + shift_val;

figure;
scatter(x1, y1, 15, 'b', 'filled');
hold on;
scatter(x2, y2, 15, 'r', 'filled');
axis equal;
grid on;
hold off;

%%
Nsrc = 50;
flanker = 1;
Ndistr = 2;

flanker = flanker*Fs;

% set filters
[b,a] = butter(4,[8,12]/(Fs/2)); % alpha band for sources
[bn,an] = butter(4,[1,35]/(Fs/2)); % for sensor noise
[be, ae] = butter(4, 0.5 / (Fs / 2), 'low'); % for envelopes

% init forward model
Gx = G(:,1:3:end);  
Gy = G(:,2:3:end);  
Gz = G(:,3:3:end);  
[Nsens, Nsites] = size(Gx);

% Create random sources with random direction
GA = zeros(Nsens, Nsrc);
src_indsA = randperm(Nsites, Nsrc);

for i = 1:Nsrc
    src_idx = src_indsA(i);
    r = rand(3,1)*2 - 1;
    r = r / norm(r);          
    GA(:,i) = Gx(:,src_idx)*r(1) + Gy(:,src_idx)*r(2) + Gz(:,src_idx)*r(3);
end

% Generate source timeseries
S = filtfilt(b,a,randn(Nsrc,N+2*flanker)')';
S = S(:,flanker+1:end-flanker);

M = filtfilt(be,ae,randn(Nsrc,N+2*flanker)')';
M = M(:,flanker+1:end-flanker);

for k = Ndistr+1:Nsrc    
    m = M(k,:); 
    m = (m - mean(m)) / std(m);
    M(k,:) = m - min(m) + eps;     
end
M(1,:) = x1;
M(2,:) = x2;

% Create random envelopes for every source
for k = 1:Nsrc
    S(k,:) = (S(k,:) - mean(S(k,:))) / std(S(k,:));
    env = abs(hilbert(S(k,:)')');
    S(k,:) = S(k,:) ./ (env + eps);
        
    S(k,:) = S(k,:) .* M(k,:);
    S(k,:) = S(k,:) - mean(S(k,:));
end

% generate sensor data
X_s = GA(:,1:Ndistr) * S(1:Ndistr,:);
X_bg = GA(:,Ndistr+1:end) * S(Ndistr+1:end,:);

% generate white noise
X_n = filtfilt(bn,an,randn(Nsens,N+2*flanker)')';
X_n = X_n(:,flanker+1:end-flanker);
X_n = X_n - mean(X_n,2);
X_n = X_n ./ std(X_n,0,2);

%%
env = abs(hilbert(S(1,:)));
plot(env)

%%
SNR = 5;

X = SNR * X_s + X_bg + X_n;

%%
Ws = 0.5;
Ss = 0.25;
X_epochs = epoch_data(X,Fs,Ws,Ss);

X_covs = [];
for ep_i=1:size(X_epochs,3)
    X_covs(:,:,ep_i) = cov(X_epochs(:,:,ep_i));
end

%%