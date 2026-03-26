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

Nsrc = 101;
Ndistr = 1;

flanker = 1;
nTr = 850;
nTe = 850;
Ts = nTr + nTe;
Fs = 250;

nMC = 10;

SNR_range = 10.^(-1.4:0.2:1);
nSNR = length(SNR_range);

corr_gl_train = [];
corr_gl_test = [];
corr_lcl_train = [];
corr_lcl_test = [];
patcorr = [];
for mc_idx = 1:nMC

    disp(mc_idx)

    [X_s, X_bg, X_n, z, GA, S] = generate_distributed_sources( ...
        G, Nsrc, Ndistr, flanker, Ts, Fs);

    for snr_idx = 1:nSNR
        
        % =================
        % Data generation
        % =================
        SNR = SNR_range(snr_idx);
        
        X = SNR*X_s + X_bg + 0.1 * X_n / norm(X_s,'fro');

        Ws = 1;
        Ss = 1;

        X_epo = epoch_data(X',Fs,Ws,Ss);

        z_epo = epoch_data(z(1,:)',Fs,Ws,Ss);
        z_epo = squeeze(mean(z_epo,1));

        % =================
        % train / test split
        % =================
        X_epo_train = X_epo(:,:,1:nTr);
        z_epo_train = z_epo(1:nTr);

        X_epo_test = X_epo(:,:,nTr+1:nTr+nTe);
        z_epo_test = z_epo(nTr+1:nTr+nTe);
        
        % =================
        % covariance matrices
        % =================
        nChan = size(X_epo_test,2);
        nFeat = nChan*(nChan+1)/2;
        
        Covs_train = zeros(nChan,nChan,nTr);
        
        Feat_train = zeros(nFeat,nTr);
        for ep_idx = 1:nTr
            C = cov(X_epo_train(:,:,ep_idx));
            Covs_train(:,:,ep_idx) = C;
            Feat_train(:,ep_idx) = cov2upper(C);
        end
        
        Covs_test = zeros(nChan,nChan,nTe);
        
        Feat_test  = zeros(nFeat,nTe);
        for ep_idx = 1:nTe
            C = cov(X_epo_test(:,:,ep_idx));
            Covs_test(:,:,ep_idx) = C;
            Feat_test(:,ep_idx) = cov2upper(C);
        end
        
        % ===== eSPoC =====
        
        [W, A, Vf, ~, corrs] = espoc(X_epo_train,z_epo_train);
        
        w = W(:,1);
        
        env_test = zeros(nTe,1);
        
        for ep_idx = 1:nTe
            env_test(ep_idx) = w' * Covs_test(:,:,ep_idx) * w;
        end
        
        gl_env_train = Vf(:,1)' * Feat_train;
        gl_env_test  = Vf(:,1)' * Feat_test;
        
        corr_gl_train(mc_idx,snr_idx) = corr(gl_env_train',z_epo_train');
        corr_gl_test(mc_idx,snr_idx)  = corr(gl_env_test',z_epo_test');
        
        corr_lcl_train(mc_idx,snr_idx) = corrs(1);
        corr_lcl_test(mc_idx,snr_idx)  = corr(env_test,z_epo_test');
        
        patcorr(mc_idx,snr_idx) = abs(corr(GA(:,1),A(:,1)));
    end
end

%% Вычисление статистики
% =================
% Statistics
% =================

% Means
mean_gl_train = mean(corr_gl_train,1);
mean_gl_test  = mean(corr_gl_test,1);

mean_lcl_train = mean(corr_lcl_train,1);
mean_lcl_test  = mean(corr_lcl_test,1);

mean_pat = mean(patcorr,1);

% 95% CI
ci_gl_train = 1.96 * std(corr_gl_train,0,1) / sqrt(nMC);
ci_gl_test  = 1.96 * std(corr_gl_test,0,1)  / sqrt(nMC);

ci_lcl_train = 1.96 * std(corr_lcl_train,0,1) / sqrt(nMC);
ci_lcl_test  = 1.96 * std(corr_lcl_test,0,1)  / sqrt(nMC);

ci_pat = 1.96 * std(patcorr,0,1) / sqrt(nMC);


% =================
% Plot
% =================

x = SNR_range;

xticks_vals = 10.^[-1 -0.4 0 0.4 1];
xticks_lbls = {'10^{-1}','10^{-0.4}','10^{0}','10^{0.4}','10^{1}'};

figure('Position',[100 100 1000 400]);

% =================
% Envelope correlation
% =================
subplot(1,2,1)
hold on

colors = [
0 0.3 1
0 0.3 1
1 0 0
1 0 0
];

styles = {'--','-','--','-'};

means = {
mean_gl_train
mean_gl_test
mean_lcl_train
mean_lcl_test
};

cis = {
ci_gl_train
ci_gl_test
ci_lcl_train
ci_lcl_test
};

labels = {
'Global train'
'Global test'
'Local train'
'Local test'
};

for i = 1:4
    
    y  = means{i};
    ci = cis{i};
    
    fill([x fliplr(x)], ...
         [y-ci fliplr(y+ci)], ...
         colors(i,:), ...
         'FaceAlpha',0.2, ...
         'EdgeColor','none', ...
         'HandleVisibility','off');
     
    semilogx(x,y, ...
        'Color',colors(i,:), ...
        'LineStyle',styles{i}, ...
        'LineWidth',2)

end

title('Envelope Correlation')
xlabel('SNR')
ylabel('Correlation')
ylim([0 1])

grid on
ax = gca;
ax.XScale = 'log';
ax.XMinorGrid = 'on';
ax.XTick = xticks_vals;
ax.XTickLabel = xticks_lbls;

legend(labels,'Location','southeast')


% =================
% Pattern correlation
% =================
subplot(1,2,2)
hold on

fill([x fliplr(x)], ...
     [mean_pat-ci_pat fliplr(mean_pat+ci_pat)], ...
     [1 0 0], ...
     'FaceAlpha',0.25, ...
     'EdgeColor','none', ...
     'HandleVisibility','off');

semilogx(x,mean_pat,'Color',[1 0 0],'LineWidth',2)

title('Pattern Correlation')
xlabel('SNR')
ylabel('Correlation')
ylim([0 1])

grid on
ax = gca;
ax.XScale = 'log';
ax.XMinorGrid = 'on';
ax.XTick = xticks_vals;
ax.XTickLabel = xticks_lbls;

legend({'Pattern'},'Location','southeast')

