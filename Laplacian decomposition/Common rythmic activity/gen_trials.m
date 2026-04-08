function [Xtrials, Xraw, tm_snr, targetA] = gen_trials( ...
    G, NConstSrc, Ntg, flanker, TrLeSe, ...
    Fs, NTr, NLclSrc, SNR)
    
    [Nsens, ~] = size(G); 
    [be, ae] = butter(2, 0.5 / (Fs / 2), 'low'); 
    tm = filtfilt(be,ae,randn(Ntg,TrLeSe*Fs + 2*flanker*Fs)')';
    tm = tm(:,flanker*Fs+1:end-flanker*Fs);
    tm = (tm - mean(tm,2)) ./ std(tm,[],2);
    
    [~, XBgConst] = generate_distributed_sources( ...
        G, NConstSrc, ...
        0, flanker, TrLeSe*NTr, Fs, false);
        
    nSamplesTrial = TrLeSe * Fs;
    XBgConst = reshape(XBgConst, [Nsens, nSamplesTrial, NTr]);
    noise_scale = norm(XBgConst(:,:,1), 'fro');
    
    Gx = G(:,1:3:end);  
    Gy = G(:,2:3:end);  
    Gz = G(:,3:3:end);  
    Nsites = size(Gx, 2);
    
    GA_target = zeros(Nsens, Ntg);
    src_indsA = randperm(Nsites, Ntg);
    
    for i = 1:Ntg
        src_idx = src_indsA(i);
        r = rand(3,1)*2 - 1;
        r = r / norm(r);          
        GA_target(:,i) = Gx(:,src_idx)*r(1) + Gy(:,src_idx)*r(2) + Gz(:,src_idx)*r(3);
    end
    % -----------------------------------------------------------------
    
    Xtrials = zeros(nSamplesTrial, Nsens, NTr);
    Xraw = zeros(Nsens, nSamplesTrial * NTr);
    targetA = zeros(NTr, Nsens, Ntg);
    
    for trial_idx=1:NTr
        tm_snr = tm * SNR;
        tm_snr = tm_snr - min(tm_snr, [], 2) + eps;
        
        [XS, XBgLcl, XN, GA] = generate_distributed_sources( ...
            G, NLclSrc, ...
            Ntg, flanker, TrLeSe, Fs, tm_snr, GA_target);
        
        targetA(trial_idx,:,:) = GA(:,1:Ntg);
        
        X = XS + XBgLcl + XBgConst(:,:,trial_idx) + 0.1 * XN * noise_scale / norm(XN,'fro');
        
        Xtrials(:,:,trial_idx) = X';
        
        start_idx = (trial_idx - 1) * nSamplesTrial + 1;
        end_idx = trial_idx * nSamplesTrial;
        Xraw(:, start_idx:end_idx) = X; 
    end
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [X_s, X_bg, X_n, GA] = generate_distributed_sources(G, Nsrc, ...
    Ndistr, flanker, Ts, Fs, target_modulator, GA_in)
    
    N = Ts*Fs;
    flanker = flanker*Fs;
    [b,a] = butter(4,[8,12]/(Fs/2));
    [bn,an] = butter(4,[1,35]/(Fs/2)); 
    [bem,aem] = butter(4, 0.5 / (Fs / 2), 'low');
    [ben,aen] = butter(5,5/(Fs/2));
    
    Gx = G(:,1:3:end);  
    Gy = G(:,2:3:end);  
    Gz = G(:,3:3:end);  
    [Nsens, Nsites] = size(Gx);
    GA = zeros(Nsens, Nsrc);
    
    if nargin < 8 || isempty(GA_in)
        src_inds_tgt = randperm(Nsites, Ndistr);
        for i = 1:Ndistr
            src_idx = src_inds_tgt(i);
            r = rand(3,1)*2 - 1;
            r = r / norm(r);          
            GA(:,i) = Gx(:,src_idx)*r(1) + Gy(:,src_idx)*r(2) + Gz(:,src_idx)*r(3);
        end
    else
        GA(:, 1:Ndistr) = GA_in; 
    end
    
    N_local_bg = Nsrc - Ndistr;
    if N_local_bg > 0
        src_inds_bg = randperm(Nsites, N_local_bg);
        for i = 1:N_local_bg
            src_idx = src_inds_bg(i);
            r = rand(3,1)*2 - 1;
            r = r / norm(r);          
            GA(:, Ndistr + i) = Gx(:,src_idx)*r(1) + Gy(:,src_idx)*r(2) + Gz(:,src_idx)*r(3);
        end
    end
    % -----------------------------------------------------------------
    
    S = filtfilt(b,a,randn(Nsrc,N+2*flanker)')';
    S = S(:,flanker+1:end-flanker);
    M = filtfilt(bem,aem,randn(Nsrc,N+2*flanker)')';
    M = M(:,flanker+1:end-flanker);
    
    if isnumeric(target_modulator) && Ndistr > 0
        M(1:Ndistr,:) = target_modulator;
    end
    
    for k = 1:Ndistr    
        m = M(k,:); 
        env_n = filtfilt(ben,aen,randn(1,N+2*flanker));
        env_n = env_n(flanker+1:end-flanker);
        env_n = env_n ./ norm(env_n);
        m = m + 0 * norm(m) * env_n;
        M(k,:) = m - min(m) + eps;     
    end
    
    for k = Ndistr+1:Nsrc    
        m = M(k,:); 
        m = (m - mean(m)) / std(m);
        env_n = filtfilt(ben,aen,randn(1,N+2*flanker));
        env_n = env_n(flanker+1:end-flanker);
        env_n = env_n ./ norm(env_n);
        m = m + 0.1 * norm(m) * env_n;
        M(k,:) = m - min(m) + eps;     
    end
    
    for k = 1:Nsrc
        S(k,:) = (S(k,:) - mean(S(k,:))) / std(S(k,:));
        env = abs(hilbert(S(k,:)')');
        S(k,:) = S(k,:) ./ (env + eps); 
            
        S(k,:) = S(k,:) .* M(k,:);
        S(k,:) = S(k,:) - mean(S(k,:));
    end
    
    X_s = GA(:,1:Ndistr) * S(1:Ndistr,:);
    X_bg = GA(:,Ndistr+1:end) * S(Ndistr+1:end,:);
    X_n = filtfilt(bn,an,randn(Nsens,N+2*flanker)')';
    X_n = X_n(:,flanker+1:end-flanker);
    X_n = X_n - mean(X_n,2);
    X_n = X_n ./ std(X_n,0,2);
end
