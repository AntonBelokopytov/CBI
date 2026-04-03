function [X_s, X_bg, X_n, GA, S] = generate_distributed_sources_envCorrCA(G, Nsrc, ...
    Ndistr, flanker, Ts, Fs, targetA, target_modulator)

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
src_indsA = randperm(Nsites, Nsrc);

for i = 1:Nsrc
    src_idx = src_indsA(i);
    r = rand(3,1)*2 - 1;
    r = r / norm(r);          
    GA(:,i) = Gx(:,src_idx)*r(1) + Gy(:,src_idx)*r(2) + Gz(:,src_idx)*r(3);
end
GA(:,1:Ndistr) = targetA;

% Generate source timeseries
S = filtfilt(b,a,randn(Nsrc,N+2*flanker)')';
S = S(:,flanker+1:end-flanker);

M = filtfilt(be,ae,randn(Nsrc,N+2*flanker)')';
M = M(:,flanker+1:end-flanker);
M(1:Ndistr,:) = target_modulator;

for k = Ndistr+1:Nsrc    
    m = M(k,:); 
    m = (m - mean(m)) / std(m);
    M(k,:) = m - min(m) + eps;     
end

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

end
