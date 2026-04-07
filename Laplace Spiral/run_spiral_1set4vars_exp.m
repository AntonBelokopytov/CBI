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
f = 0.5;
Nc = 1.5; 
noise_level = 0.1; 
Ts = 100;
Fs = 250;
N = Ts*Fs;

t = linspace(0, 2*pi*f*Nc, N); 
A = linspace(0, 6, N);

x1 = A .* cos(t + phi1) + noise_level * randn(size(A));
y1 = A .* sin(t + phi1) + noise_level * randn(size(A));

x2 = A .* cos(t + phi2) + noise_level * randn(size(A));
y2 = A .* sin(t + phi2) + noise_level * randn(size(A));

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
Nsrc = 10;
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
M(3,:) = y1;
M(4,:) = y2;

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
figure
env = abs(hilbert(S(1,:)));
plot(env); hold on
env = abs(hilbert(S(2,:)));
plot(env)

figure
env = abs(hilbert(S(3,:)));
plot(env); hold on
env = abs(hilbert(S(4,:)));
plot(env)

%%
SNR = 20;

X = SNR * X_s + X_bg + X_n;

Ws = 2;
Ss = 0.2;
X_epochs = epoch_data(X',Fs,Ws,Ss);

X_covs = [];
for ep_i=1:size(X_epochs,3)
    X_covs(:,:,ep_i) = cov(X_epochs(:,:,ep_i));
end

Dists = zeros(size(X_covs,3));

for ep_i=1:size(X_covs,3)-1
    ep_i
    for ep_j=ep_i+1:size(X_covs,3)
        C1 = X_covs(:,:,ep_i);
        C2 = X_covs(:,:,ep_j);
        d = distance_riemann(C1,C2);
        Dists(ep_i,ep_j) = d;
    end
end
Dists = Dists + Dists';

%%
clear u
u = UMAP("n_neighbors",50,"n_components",2,'metric','precomputed');
R = u.fit_transform(Dists);

%%
figure;
scatter(R(:, 1), R(:, 2), 15, 'b', 'filled');
axis equal;
grid on;
hold off;

%%
% W = 1 ./ Dists;
% sigma = median(Dists(:)); 
% t = 1 * sigma^2; 
% n_neighbors = 100; 
% W_knn = zeros(size(Dists));
% for i = 1:size(Dists, 1)
%     [~, sorted_idx] = sort(Dists(i, :), 'ascend');
%     neighbors_idx = sorted_idx(2 : n_neighbors + 1);    
%     W_knn(i, neighbors_idx) = exp(-(Dists(i, neighbors_idx).^2) / t);
% end
% W = max(W_knn, W_knn');

k = 100; 
W_full = 1 ./ (Dists + eps); 
W_full(logical(eye(size(W_full)))) = 0;

knn_mask = false(size(Dists));
for i = 1:size(Dists, 1)
    [~, sorted_idx] = sort(Dists(i, :), 'ascend');

    neighbors_idx = sorted_idx(2 : k + 1);    
    knn_mask(i, neighbors_idx) = true;
end
knn_mask = knn_mask | knn_mask'; 
W = W_full .* knn_mask;

W(logical(eye(size(W)))) = 0;
D = diag(sum(W,2));
L = D - W;

d_vec = diag(D); 

Dsqrt = diag(sqrt(d_vec)); 

d_inv_vec = 1 ./ sqrt(d_vec);
d_inv_vec(isinf(d_inv_vec)) = 0; 
Dinv = diag(d_inv_vec);

[V, S_l] = eig(Dinv * L * Dinv);

eig_vals = diag(S_l);

[eig_vals, sort_idx] = sort(eig_vals);
V = V(:, sort_idx);

Y = Dinv * V;
Y = (Y - mean(Y,1)) ./ std(Y,[],1);

figure
plot(Y(:, 2)); hold on
plot(Y(:, 3));

figure
plot(Y(:, 4)); hold on
plot(Y(:, 5));


figure;
scatter(Y(:, 2), Y(:, 3), 15, 'b', 'filled');
hold on;
scatter(Y(:, 4), Y(:, 5), 15, 'r', 'filled');
axis equal;
grid on;
hold off;

