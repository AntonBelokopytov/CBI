close all
clear
clc

ft_path = 'C:\Users\ansbel\Documents\2Git\fieldtrip';

if ~exist('ft_defaults','file')
    addpath(ft_path);
end

ft_defaults;

%%
methods = {@espoc, @mspoc, @spoc};

G = load('MNE_EEG_FWD_TRPL.mat').MNE_EEG_FWD_TRPL;

Nsrc = 102;
Ndistr = 2;

flanker = 1;
Ts = 850;
Fs = 250;

nMC = 100;

SNR_range = 10.^(-1.4:0.2:1);
nSNR = length(SNR_range);

filcorr = zeros(nMC,nSNR,numel(methods),Ndistr);
patcorr = zeros(nMC,nSNR,numel(methods),Ndistr);
z_unmixing_corr = zeros(nMC,nSNR,2,Ndistr);

for mc_idx = 1:nMC

    [X_s, X_bg, X_n, Zinit, GA, S] = generate_distributed_sources( ...
        G, Nsrc, Ndistr, flanker, Ts, Fs);

    Zinit = Zinit(1:Ndistr,:);

    Az = randn(Ndistr,Ndistr);
    Zmixed  = Az' * Zinit; 

    disp(mc_idx)

    filcorr_local = zeros(nSNR,numel(methods),Ndistr);
    patcorr_local = zeros(nSNR,numel(methods),Ndistr);
    z_unmixing_corr_local = zeros(nSNR,2,Ndistr);
    parfor snr_idx = 1:nSNR
        % =================
        % Data generation
        % =================

        SNR = SNR_range(snr_idx);

        X = SNR*X_s + X_bg + 0.1 * X_n / norm(X_s,'fro');

        X_epo = epoch_data(X',Fs,1,1);

        z_epo_mixed = epoch_data(Zmixed',Fs,1,1);
        z_epo_mixed = squeeze(mean(z_epo_mixed,1));

        z_epo_init = epoch_data(Zinit',Fs,1,1);
        z_epo_init = squeeze(mean(z_epo_init,1));

        % =================
        % Train / test split
        % =================

        X_epo_train = X_epo(:,:,1:250);
        z_epo_train_mixed = z_epo_mixed(:,1:250);
        z_epo_test_mixed = z_epo_mixed(:,251:850);

        X_epo_test = X_epo(:,:,251:850);
        z_epo_test_init = z_epo_init(:,251:850);

        nTrain = size(X_epo_train,3);
        nTest = size(X_epo_test,3);
        nChan = size(X_epo_test,2);

        % Covs_train = zeros(nChan,nChan,nTrain);
        % 
        % for ep_idx = 1:nTrain
        %     Covs_train(:,:,ep_idx) = cov(X_epo_train(:,:,ep_idx));
        % end

        Covs_test = zeros(nChan,nChan,nTest);

        for ep_idx = 1:nTest
            Covs_test(:,:,ep_idx) = cov(X_epo_test(:,:,ep_idx));
        end

        filcorr_snr = zeros(numel(methods),Ndistr);
        patcorr_snr = zeros(numel(methods),Ndistr);
        z_unmixing_corr_snr = zeros(2,Ndistr);
        for m_idx = 1:numel(methods)
            w = [];
            a = [];
            
            switch m_idx

                case 1 % espoc

                    [W,A,~,Vz] = espoc(X_epo_train,z_epo_train_mixed);

                    w = [W(1,:,1)' W(1,:,end)' W(2,:,1)' W(2,:,end)'];
                    a = [A(1,:,1)' A(1,:,end)' A(2,:,1)' A(2,:,end)'];

                case 2 % mspoc

                    [W,Vz,~,A] = mspoc(X_epo_train,z_epo_train_mixed);

                    w = W;
                    a = A;

                case 3 % spoc

                    [W1,A1] = spoc(X_epo_train,z_epo_train_mixed(1,:));
                    [W2,A2] = spoc(X_epo_train,z_epo_train_mixed(2,:));

                    w = [W1(:,1) W1(:,2) W1(:,end-1) W1(:,end) ...
                         W2(:,1) W2(:,2) W2(:,end-1) W2(:,end)];

                    a = [A1(:,1) A1(:,2) A1(:,end-1) A1(:,end) ...
                         A2(:,1) A2(:,2) A2(:,end-1) A2(:,end)];

                case 4 % spoc_r2

                    [W1,A1] = spoc_r2(X_epo_train,z_epo_train_mixed(1,:));
                    [W2,A2] = spoc_r2(X_epo_train,z_epo_train_mixed(2,:));
                    
                    w = [W1(:,1:4) W2(:,1:4)];
                    a = [A1(:,1:4) A2(:,1:4)];

            end
            
            [f_idx,f_corr] = select_filters(w,Covs_test,z_epo_test_init);

            filcorr_snr(m_idx,:) = f_corr;
            patcorr_snr(m_idx,:) = diag(abs(corr(GA(:,1:Ndistr),a(:,f_idx))));
            
            if m_idx < 3
            switch m_idx
                case 1 % espoc

                    if f_idx(1) < 3
                        Vz1 = Vz(:,1);
                        Vz2 = Vz(:,2);
                    else
                        Vz2 = Vz(:,1);
                        Vz1 = Vz(:,2);
                    end
                    
                case 2 % mspoc

                    if f_idx(1) == 1
                        Vz1 = Vz(:,1);
                        Vz2 = Vz(:,2);
                    else
                        Vz2 = Vz(:,1);
                        Vz1 = Vz(:,2);
                    end                
            end
            z_epo_test_unmixed1 = Vz1' * z_epo_test_mixed;
            zcorr1 = abs(corr(z_epo_test_unmixed1',z_epo_test_init(1,:)'));

            z_epo_test_unmixed2 = Vz2' * z_epo_test_mixed;
            zcorr2 = abs(corr(z_epo_test_unmixed2',z_epo_test_init(2,:)'));
            
            zcorrs = [zcorr1, zcorr2];
            z_unmixing_corr_snr(m_idx,:) = zcorrs;
            end
        end

        filcorr_local(snr_idx,:,:) = filcorr_snr;
        patcorr_local(snr_idx,:,:) = patcorr_snr;
        z_unmixing_corr_local(snr_idx,:,:) = z_unmixing_corr_snr;

    end

    filcorr(mc_idx,:,:,:) = filcorr_local;
    patcorr(mc_idx,:,:,:) = patcorr_local;
    z_unmixing_corr(mc_idx,:,:,:) = z_unmixing_corr_local;
end

%%
nMC = mc_idx - 1;
filcorr = filcorr(1:nMC,:,:,:);
patcorr = patcorr(1:nMC,:,:,:);
z_unmixing_corr = z_unmixing_corr(1:nMC,:,:,:);

%% =================
% Statistics
% =================

nMethods = numel(methods);

% Mean
mean_filt = squeeze(mean(filcorr,1));    % (nSNR × nMethods × Ndistr)
mean_pat  = squeeze(mean(patcorr,1));

% CI
ci_filt = squeeze(1.96 * std(filcorr,0,1) / sqrt(nMC));
ci_pat  = squeeze(1.96 * std(patcorr,0,1) / sqrt(nMC));

% =================
% Plot
% =================

x = SNR_range;

xticks_vals = 10.^[-1 -0.4 0 0.4 1];
xticks_lbls = {'10^{-1}','10^{-0.4}','10^{0}','10^{0.4}','10^{1}'};

colors = [
0.3 0.6 1
0.3 0.8 0.3
1 0.4 0.4
0.7 0.3 0.9
];

styles = {'-','-','-','--'};

labels = {'eSPoC','mSPoC','SPoC','SPoC\_r2'};

figure('Position',[100 100 1200 800])

for src_idx = 1:Ndistr

    % Envelope correlation
    subplot(Ndistr,2,(src_idx-1)*2+1)
    hold on

    for m = 1:nMethods
        
        y  = mean_filt(:,m,src_idx);
        ci = ci_filt(:,m,src_idx);

        errorbar(x, y, ci, ...
            'LineStyle','none', ...
            'Color',colors(m,:), ...
            'CapSize',6, ...
            'HandleVisibility','off');

        semilogx(x,y,'Color',colors(m,:), ...
            'LineWidth',2, ...
            'LineStyle',styles{m});

    end

    title(['Envelope corr – source ' num2str(src_idx)])
    ylabel('Correlation')
    ylim([0 1])
    grid on

    ax = gca;
    ax.XScale = 'log';
    ax.XMinorGrid = 'on';
    ax.XTick = xticks_vals;
    ax.XTickLabel = xticks_lbls;

    if src_idx==Ndistr
        xlabel('SNR')
    end

    if src_idx==1
        legend(labels,'Location','southeast')
    end


    % Pattern correlation
    subplot(Ndistr,2,(src_idx-1)*2+2)
    hold on

    for m = 1:nMethods
        
        y  = mean_pat(:,m,src_idx);
        ci = ci_pat(:,m,src_idx);

        errorbar(x, y, ci, ...
            'LineStyle','none', ...
            'Color',colors(m,:), ...
            'CapSize',6, ...
            'HandleVisibility','off');

        semilogx(x,y,'Color',colors(m,:), ...
            'LineWidth',2, ...
            'LineStyle',styles{m});

    end

    title(['Pattern corr – source ' num2str(src_idx)])
    ylim([0 1])
    grid on

    ax = gca;
    ax.XScale = 'log';
    ax.XMinorGrid = 'on';
    ax.XTick = xticks_vals;
    ax.XTickLabel = xticks_lbls;

    if src_idx==Ndistr
        xlabel('SNR')
    end

end

%% =================
% Statistics for external variable recovery
% =================

mean_z = squeeze(mean(z_unmixing_corr,1));   % (nSNR × method × source)
ci_z   = squeeze(1.96 * std(z_unmixing_corr,0,1) / sqrt(nMC));

% =================
% Plot regressor recovery
% =================

colors = [
0.3 0.6 1
0.3 0.8 0.3
];

labels = {'eSPoC','mSPoC'};

figure('Position',[200 200 900 400])

for src_idx = 1:Ndistr
    
    subplot(1,Ndistr,src_idx)
    hold on
    
    for m = 1:2
        
        y  = mean_z(:,m,src_idx);
        ci = ci_z(:,m,src_idx);
        
        % доверительный интервал
        errorbar(x, y, ci, ...
            'LineStyle','none', ...
            'Color',colors(m,:), ...
            'CapSize',6, ...
            'HandleVisibility','off');
        
        % основная линия
        semilogx(x, y, ...
            'Color',colors(m,:), ...
            'LineWidth',2);
        
    end
    
    title(['Regressor recovery – source ' num2str(src_idx)])
    ylabel('Correlation')
    xlabel('SNR')
    ylim([0 1])
    grid on
    
    ax = gca;
    ax.XScale = 'log';
    ax.XMinorGrid = 'on';
    ax.XTick = xticks_vals;
    ax.XTickLabel = xticks_lbls;
    
    legend(labels,'Location','southeast')
    
end
