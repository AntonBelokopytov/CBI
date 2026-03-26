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
methods = {@espoc, @spoc, @spoc_r2};

G = load('MNE_EEG_FWD_TRPL.mat').MNE_EEG_FWD_TRPL;

Nsrc = 101;
Ndistr = 1;

flanker = 1;
Ts = 850;
Fs = 250;

nMC = 500;

SNR_range = 10.^(-1.4:0.2:1);
nSNR = length(SNR_range);

filcorr = zeros(nMC,nSNR,3);
patcorr = zeros(nMC,nSNR,3);

for mc_idx = 1:nMC

    disp(mc_idx)

    [X_s, X_bg, X_n, z, GA, S] = generate_distributed_sources( ...
        G, Nsrc, Ndistr, flanker, Ts, Fs);

    filcorr_local = zeros(nSNR,3);
    patcorr_local = zeros(nSNR,3);

    parfor snr_idx = 1:nSNR
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
        X_epo_train = X_epo(:,:,1:250);
        z_epo_train = z_epo(1:250);

        X_epo_test = X_epo(:,:,251:250+600);
        z_epo_test = z_epo(251:250+600);

        % =================
        % covariance matrices
        % =================
        nTest = size(X_epo_test,3);
        nChan = size(X_epo_test,2);

        Covs_test = zeros(nChan,nChan,nTest);

        for ep_idx = 1:nTest
            Covs_test(:,:,ep_idx) = cov(X_epo_test(:,:,ep_idx));
        end

        % =================
        % methods
        % =================
        for m_idx = 1:3

            alg = methods{m_idx};

            [W,A] = alg(X_epo_train,z_epo_train);

            w = W(:,1);

            env = zeros(nTest,1);

            for ep_idx = 1:nTest
                env(ep_idx) = w' * Covs_test(:,:,ep_idx) * w;
            end

            filcorr_local(snr_idx,m_idx) = corr(env(:),z_epo_test(:))
            patcorr_local(snr_idx,m_idx) = abs(corr(A(:,1),GA(:,1)))

        end
    end

    filcorr(mc_idx,:,:) = filcorr_local;
    patcorr(mc_idx,:,:) = patcorr_local;

end

%%
nMC = mc_idx - 1;
filcorr = filcorr(1:nMC,:,:);
patcorr = patcorr(1:nMC,:,:);

%% Вычисление статистики
% =================
% Means
% =================
mean_filt = squeeze(mean(filcorr,1));   % (nSNR x 3)
mean_pat  = squeeze(mean(patcorr,1));

% =================
% 95% CI
% =================
ci_filt = squeeze(1.96 * std(filcorr,0,1) / sqrt(nMC));
ci_pat  = squeeze(1.96 * std(patcorr,0,1) / sqrt(nMC));

% =================
% Plot
% =================
x = SNR_range;

xticks_vals = 10.^[-1 -0.4 0 0.4 1];
xticks_lbls = {'10^{-1}','10^{-0.4}','10^{0}','10^{0.4}','10^{1}'};

figure('Position',[100 100 900 400]);

colors = [0 0 1
          1 0 0
          0 0.7 0];

labels = {'eSPoC', 'SPoC','SPoC\_r2'};

% Envelope correlation
subplot(1,2,1)
hold on

for m = 1:3
    
    y  = mean_filt(:,m)';
    ci = ci_filt(:,m)';
    
    fill([x fliplr(x)], ...
         [y-ci fliplr(y+ci)], ...
         colors(m,:), ...
         'FaceAlpha',0.25, ...
         'EdgeColor','none', ...
         'HandleVisibility','off');   
    
    semilogx(x,y,'Color',colors(m,:),'LineWidth',2)

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


% Pattern correlation
subplot(1,2,2)
hold on

for m = 1:3
    
    y  = mean_pat(:,m)';
    ci = ci_pat(:,m)';
    
    fill([x fliplr(x)], ...
         [y-ci fliplr(y+ci)], ...
         colors(m,:), ...
         'FaceAlpha',0.25, ...
         'EdgeColor','none');
    
    semilogx(x,y,'Color',colors(m,:),'LineWidth',2)

end

title('Pattern Correlation')
xlabel('SNR')
ylim([0 1])

grid on
ax = gca;
ax.XScale = 'log';
ax.XMinorGrid = 'on';
ax.XTick = xticks_vals;
ax.XTickLabel = xticks_lbls;

%%


