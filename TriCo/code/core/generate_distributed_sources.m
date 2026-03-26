function [X_s, X_bg, X_n, Z, GA, S] = generate_distributed_sources(G, Nsrc, Ndistr, flanker, Ts, Fs)

N = Ts*Fs;
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
src_indsA = randperm(Nsites, Nsrc); % Чуть безопаснее, если Nsites < Nsrc

for i = 1:Nsrc
    src_idx = src_indsA(i);
    r = randn(3,1);
    r = r / norm(r);          
    GA(:,i) = Gx(:,src_idx)*r(1) + Gy(:,src_idx)*r(2) + Gz(:,src_idx)*r(3);
end

% Generate source timeseries
S = filtfilt(b,a,randn(Nsrc,N+2*flanker)')';
S = S(:,flanker+1:end-flanker);

M = filtfilt(be,ae,randn(Nsrc,N+2*flanker)')';
M = M(:,flanker+1:end-flanker);

Z = zeros(Nsrc, N);

% Create random envelopes for every source
for k = 1:Nsrc
    S(k,:) = (S(k,:) - mean(S(k,:))) / std(S(k,:));
    env = abs(hilbert(S(k,:)')');
    S(k,:) = S(k,:) ./ (env + eps);
    
    m = M(k,:); 
    m = (m - mean(m)) / std(m);
    m = m - min(m) + eps; 
    m = m / mean(m);
    
    S(k,:) = S(k,:) .* m;
    S(k,:) = (S(k,:) - mean(S(k,:))) / std(S(k,:));
    
    Z(k,:) = m.^2; 
end

% generate sensor data
X_s = GA(:,1:Ndistr) * S(1:Ndistr,:);
X_bg = GA(:,Ndistr+1:end) * S(Ndistr+1:end,:);

% generate white noise
X_n = filtfilt(bn,an,randn(Nsens,N+2*flanker)')';
X_n = X_n(:,flanker+1:end-flanker);
X_n = X_n - mean(X_n,2);
X_n = X_n ./ std(X_n,0,2);

end
