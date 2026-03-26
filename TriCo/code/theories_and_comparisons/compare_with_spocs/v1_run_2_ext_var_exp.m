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
Nsrc = 102;
Ndistr = 2;
flanker = 1;
Ts = 250;
Fs = 250;
nMC = 500;
SNR_range = 10.^(-1.4:0.2:1);
nSNR = length(SNR_range);

% Инициализация массивов (добавлен sp2)
filtcorr_esp = zeros(nMC,nSNR,Ndistr);
filtcorr_sp  = zeros(nMC,nSNR,Ndistr);
filtcorr_sp2 = zeros(nMC,nSNR,Ndistr);
filtcorr_msp = zeros(nMC,nSNR,Ndistr);

patcorr_esp  = zeros(nMC,nSNR,Ndistr);
patcorr_sp   = zeros(nMC,nSNR,Ndistr);
patcorr_sp2  = zeros(nMC,nSNR,Ndistr);
patcorr_msp  = zeros(nMC,nSNR,Ndistr);

parfor mc = 1:nMC
    fprintf('MC iteration: %d\n', mc);
    filtcorr_esp_mc = zeros(nSNR,Ndistr);
    filtcorr_sp_mc  = zeros(nSNR,Ndistr);
    filtcorr_sp2_mc = zeros(nSNR,Ndistr);
    filtcorr_msp_mc = zeros(nSNR,Ndistr);
    
    patcorr_esp_mc  = zeros(nSNR,Ndistr);
    patcorr_sp_mc   = zeros(nSNR,Ndistr);
    patcorr_sp2_mc  = zeros(nSNR,Ndistr);
    patcorr_msp_mc  = zeros(nSNR,Ndistr);
        
    for snr_i = 1:nSNR
        [X_s, X_bg, X_n, Z, GA, S] = generate_distributed_sources(G, Nsrc, Ndistr, flanker, Ts, Fs);
        
        Az = randn(Ndistr,Ndistr);
        z  = Az' * Z(1:Ndistr,:);

        SNR = SNR_range(snr_i);
        X = SNR*X_s + X_bg + 0.1 * X_n / norm(X_s,'fro');
        
        X_epo = epoch_data(X',Fs,1,1);
        z_epo = epoch_data(z',Fs,1,1);
        z_epo = squeeze(mean(z_epo,1));
        
        z_epo_true = epoch_data(Z(1:Ndistr,:)',Fs,1,1);
        z_epo_true = squeeze(mean(z_epo_true,1));
        
        % Извлекаем ковариации один раз для всех select_filters
        [~,n_channels,n_epochs] = size(X_epo);
        n_features = (n_channels^2-n_channels)/2+n_channels;
        
        Epochs_cov = zeros(n_channels,n_channels,n_epochs);
        for ep_idx = 1:n_epochs
            Xcov = cov(X_epo(:,:,ep_idx));
            Epochs_cov(:,:,ep_idx) = Xcov;
        end

        % ================= eSPoC =================
        [We, Ae] = espoc(X_epo, z_epo);
        Aesp = [Ae(1,:,1); Ae(1,:,end); Ae(2,:,1); Ae(2,:,end)]';
        Wesp = [We(1,:,1); We(1,:,end); We(2,:,1); We(2,:,end)]';
        [idx_e, f_e] = select_filters(Wesp, Epochs_cov, z_epo_true);
        filtcorr_esp_mc(snr_i,:) = f_e;
        patcorr_esp_mc(snr_i,:)  = diag(abs(corr(GA(:,1:Ndistr), Aesp(:,idx_e))));

        % ================= SPoC (Lambda) =================
        [Wsp1, Asp1] = spoc(X_epo, z_epo(1,:));
        [Wsp2, Asp2] = spoc(X_epo, z_epo(2,:));
        Wsp = [Wsp1(:,1:2), Wsp1(:,end-2:end), Wsp2(:,1:2), Wsp2(:,end-2:end)];
        Asp = [Asp1(:,1:2), Asp1(:,end-2:end), Asp2(:,1:2), Asp2(:,end-2:end)];
        [idx_s, f_s] = select_filters(Wsp, Epochs_cov, z_epo_true);
        filtcorr_sp_mc(snr_i,:) = f_s;
        patcorr_sp_mc(snr_i,:)  = diag(abs(corr(GA(:,1:Ndistr), Asp(:,idx_s))));

        % ================= SPoC_r2 =================
        % Запрашиваем по 2 компонента на каждый таргет для честного сравнения
        [Wsp2_1, Asp2_1] = spoc_r2(X_epo, z_epo(1,:), 'n_spoc_components', 2, 'verbose', 0);
        [Wsp2_2, Asp2_2] = spoc_r2(X_epo, z_epo(2,:), 'n_spoc_components', 2, 'verbose', 0);
        Wsp2 = [Wsp2_1, Wsp2_2];
        Asp2 = [Asp2_1, Asp2_2];
        [idx_s2, f_s2] = select_filters(Wsp2, Epochs_cov, z_epo_true);
        filtcorr_sp2_mc(snr_i,:) = f_s2;
        patcorr_sp2_mc(snr_i,:)  = diag(abs(corr(GA(:,1:Ndistr), Asp2(:,idx_s2))));

        % ================= mSPoC =================
        [Wx, ~, ~, Ax] = mspoc(X_epo, z_epo, 'n_component_sets', Ndistr, 'verbose', 0);
        Wmsp = Wx(:,1:Ndistr);
        Amsp = Ax(:,1:Ndistr);
        [idx_m, f_m] = select_filters(Wmsp, Epochs_cov, z_epo_true);
        filtcorr_msp_mc(snr_i,:) = f_m;
        patcorr_msp_mc(snr_i,:)  = diag(abs(corr(GA(:,1:Ndistr), Amsp(:,idx_m))));
    end
    filtcorr_esp(mc,:,:) = filtcorr_esp_mc;
    filtcorr_sp(mc,:,:)  = filtcorr_sp_mc;
    filtcorr_sp2(mc,:,:) = filtcorr_sp2_mc;
    filtcorr_msp(mc,:,:) = filtcorr_msp_mc;
    
    patcorr_esp(mc,:,:) = patcorr_esp_mc;
    patcorr_sp(mc,:,:)  = patcorr_sp_mc;
    patcorr_sp2(mc,:,:) = patcorr_sp2_mc;
    patcorr_msp(mc,:,:) = patcorr_msp_mc;
end

%% statistics
% Добавляем расчеты для SPoC_r2
mean_filt_esp = squeeze(mean(filtcorr_esp,1));
mean_filt_sp  = squeeze(mean(filtcorr_sp,1));
mean_filt_sp2 = squeeze(mean(filtcorr_sp2,1)); 
mean_filt_msp = squeeze(mean(filtcorr_msp,1));

mean_pat_esp = squeeze(mean(patcorr_esp,1));
mean_pat_sp  = squeeze(mean(patcorr_sp,1));
mean_pat_sp2 = squeeze(mean(patcorr_sp2,1)); 
mean_pat_msp = squeeze(mean(patcorr_msp,1));

% CI для всех методов
ci_filt_esp = 1.96*squeeze(std(filtcorr_esp,0,1))/sqrt(nMC);
ci_filt_sp  = 1.96*squeeze(std(filtcorr_sp,0,1))/sqrt(nMC);
ci_filt_sp2 = 1.96*squeeze(std(filtcorr_sp2,0,1))/sqrt(nMC); % NEW
ci_filt_msp = 1.96*squeeze(std(filtcorr_msp,0,1))/sqrt(nMC);

ci_pat_esp = 1.96*squeeze(std(patcorr_esp,0,1))/sqrt(nMC);
ci_pat_sp  = 1.96*squeeze(std(patcorr_sp,0,1))/sqrt(nMC);
ci_pat_sp2 = 1.96*squeeze(std(patcorr_sp2,0,1))/sqrt(nMC); % NEW
ci_pat_msp = 1.96*squeeze(std(patcorr_msp,0,1))/sqrt(nMC);

x = SNR_range;
xticks_vals = 10.^[-1 -0.4 0 0.4 1];
xticks_lbls = {'10^{-1}','10^{-0.4}','10^{0}','10^{0.4}','10^{1}'};

% Цвета
c_esp = [0.3 0.6 1];   % Синий
c_sp  = [1 0.4 0.4];   % Красный
c_sp2 = [0.7 0.3 0.9]; % Фиолетовый
c_msp = [0.3 0.8 0.3]; % Зеленый

figure('Position', [100 100 1200 800])

for src_idx = 1:2
    % --- ГРАФИК КОРРЕЛЯЦИИ ОГИБАЮЩИХ (ENVELOPE) ---
    subplot(2,2, src_idx*2-1); hold on
    
    % eSPoC
    plot_with_ci(x, mean_filt_esp(:,src_idx), ci_filt_esp(:,src_idx), c_esp, '-');
    % SPoC
    plot_with_ci(x, mean_filt_sp(:,src_idx), ci_filt_sp(:,src_idx), c_sp, '-');
    % SPoC_r2
    plot_with_ci(x, mean_filt_sp2(:,src_idx), ci_filt_sp2(:,src_idx), c_sp2, '--');
    % mSPoC
    plot_with_ci(x, mean_filt_msp(:,src_idx), ci_filt_msp(:,src_idx), c_msp, '-');
    
    title(['Envelope corr – source ' num2str(src_idx)])
    ylabel('Correlation'); ylim([0 1]); grid on
    set_axes_log(gca, xticks_vals, xticks_lbls);
    if src_idx == 2, xlabel('SNR'); end
    if src_idx == 1
        legend({'eSPoC CI','eSPoC','SPoC CI','SPoC','SPoC\_r2 CI','SPoC\_r2','mSPoC CI','mSPoC'},...
            'Location','southeast','FontSize',8);
    end

    % --- ГРАФИК КОРРЕЛЯЦИИ ПАТТЕРНОВ (PATTERN) ---
    subplot(2,2, src_idx*2); hold on
    
    plot_with_ci(x, mean_pat_esp(:,src_idx), ci_pat_esp(:,src_idx), c_esp, '-');
    plot_with_ci(x, mean_pat_sp(:,src_idx), ci_pat_sp(:,src_idx), c_sp, '-');
    plot_with_ci(x, mean_pat_sp2(:,src_idx), ci_pat_sp2(:,src_idx), c_sp2, '--');
    plot_with_ci(x, mean_pat_msp(:,src_idx), ci_pat_msp(:,src_idx), c_msp, '-');
    
    title(['Pattern corr – source ' num2str(src_idx)])
    ylim([0 1]); grid on
    set_axes_log(gca, xticks_vals, xticks_lbls);
    if src_idx == 2, xlabel('SNR'); end
end

% Вспомогательные функции для чистоты кода
function plot_with_ci(x, y, ci, color, style)
    fill([x fliplr(x)], [y-ci; flipud(y+ci)]', color, 'FaceAlpha', 0.15, 'EdgeColor', 'none');
    semilogx(x, y, 'Color', color, 'LineWidth', 2, 'LineStyle', style);
end

function set_axes_log(ax, ticks, labels)
    ax.XScale = 'log';
    ax.XMinorGrid = 'on';
    ax.XTick = ticks;
    ax.XTickLabel = labels;
end
