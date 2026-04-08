function [Xtrials, Xraw, tm_snr, targetA] = gen_multisub(...
    G, Ntg, flanker, TrLeSe, ...
    Fs, NTr, NLclSrc, SNR)
    
    [Nsens, ~] = size(G); 
    [be, ae] = butter(2, 0.5 / (Fs / 2), 'low'); 
    tm = filtfilt(be,ae,randn(Ntg,TrLeSe*Fs + 2*flanker*Fs)')';
    tm = tm(:,flanker*Fs+1:end-flanker*Fs);
    tm = (tm - mean(tm,2)) ./ std(tm,[],2);
    
    nSamplesTrial = TrLeSe * Fs;
    
    Xtrials = zeros(nSamplesTrial, Nsens, NTr);
    Xraw = zeros(Nsens, nSamplesTrial * NTr);
    targetA = zeros(NTr, Nsens, Ntg);
    
    for trial_idx=1:NTr
        tm_snr = tm * SNR;
        tm_snr = tm_snr - min(tm_snr, [], 2) + eps;
        
        [XS, XBgLcl, XN, GA] = generate_distributed_sources( ...
            G, NLclSrc, ...
            Ntg, flanker, TrLeSe, Fs, tm_snr);
        
        targetA(trial_idx,:,:) = GA(:,1:Ntg);
        
        noise_scale = norm(XBgLcl(:,:,1), 'fro');
        X = XS + XBgLcl + 0.1 * XN * noise_scale / norm(XN,'fro');
        
        Xtrials(:,:,trial_idx) = X';
        
        start_idx = (trial_idx - 1) * nSamplesTrial + 1;
        end_idx = trial_idx * nSamplesTrial;
        Xraw(:, start_idx:end_idx) = X; 
    end
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [X_s, X_bg, X_n, GA] = generate_distributed_sources(G, Nsrc, ...
    Ndistr, flanker, Ts, Fs, target_modulator)
    
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
    src_indsA = randperm(Nsites, Nsrc);
    
    for i = 1:Nsrc
        src_idx = src_indsA(i);
        r = rand(3,1)*2 - 1;
        r = r / norm(r);          
        GA(:,i) = Gx(:,src_idx)*r(1) + Gy(:,src_idx)*r(2) + Gz(:,src_idx)*r(3);
    end
    
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
