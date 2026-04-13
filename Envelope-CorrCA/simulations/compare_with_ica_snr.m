close all
clear
clc

ft_path = 'D:\OS(CURRENT)\scripts\2Git\fieldtrip';

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

%%  
NConstSrc = 91; 
Ntg = 1; 
flanker = 1; 
TrLeSe = 5; 
Fs = 100; 
NTr = 50; 
NLclSrc = 11;
Wsize = 1 / 8; 
Ssize = Wsize/2;

%%
snr_vec = 10.^(-1.4:0.2:1);
Nsnr = length(snr_vec);

NMC = 1000;

env_corr_spoc = zeros(Nsnr, NMC);
patt_corr_spoc = zeros(Nsnr, NMC);
env_corr_ica  = zeros(Nsnr, NMC);
patt_corr_ica = zeros(Nsnr, NMC);

for snr_i = 1:Nsnr
    snr_i

    SNR = snr_vec(snr_i);
    
    parfor mc_i = 1:NMC
        
        [Xtrials, Xraw, tm, TgPa] = gen_dat_corrca( ...
            G, NConstSrc, Ntg, flanker, TrLeSe, ...
            Fs, NTr, NLclSrc, SNR);
        
        tmraw = repmat(tm,[1,NTr]);
        
        % 
        [W, A] = env_corrca(Xtrials, Fs, Wsize, Ssize);
        
        W = squeeze(W(1,:,:)); W = [W(:,1), W(:,end)];
        A = squeeze(A(1,:,:)); A = [A(:,1), A(:,end)];
        
        env_corr = zeros(1,2);
        env_spoc = zeros(length(tmraw),2);
        
        for w_i = 1:2
            w = W(:,w_i);
            env_spoc(:,w_i) = abs(hilbert(w'*Xraw));
            env_corr(w_i) = corr(env_spoc(:,w_i),tmraw');
        end
        
        [b_corr_env, b_idx] = max(env_corr);
        
        env_corr_spoc(snr_i, mc_i) = b_corr_env;
        patt_corr_spoc(snr_i, mc_i) = abs(corr(A(:,b_idx),TgPa));
        
        % FastICA
        [Aica, Wica] = fastica(Xraw, ...
            'numOfIC', size(Xraw,1), ...
            'approach', 'symm', ...
            'g', 'tanh', ...
            'verbose', 'off');
        
        Wica = Wica';
        
        env_corr = zeros(1,size(Wica,2));
        env_ica = zeros(length(tmraw), size(Wica,2));
        
        for w_i = 1:size(Wica, 2)
            w = Wica(:,w_i);
            env_ica(:,w_i) = abs(hilbert(w'*Xraw));
            env_corr(w_i) = corr(env_ica(:,w_i), tmraw');
        end
        
        [b_corr_env, b_idx] = max(env_corr);
        
        env_corr_ica(snr_i, mc_i) = b_corr_env;
        patt_corr_ica(snr_i, mc_i) = abs(corr(Aica(:,b_idx),TgPa));
    end
end

%% Расчет статистик и построение графиков
N = NMC;

% Статистика (сокращено для краткости, используй свои расчеты mean/ci)
mean_env_spoc = mean(env_corr_spoc, 2); mean_env_ica = mean(env_corr_ica, 2);
ci_env_spoc = 1.96 * std(env_corr_spoc, 0, 2) / sqrt(N); ci_env_ica = 1.96 * std(env_corr_ica, 0, 2) / sqrt(N);
mean_patt_spoc = mean(patt_corr_spoc, 2); mean_patt_ica = mean(patt_corr_ica, 2);
ci_patt_spoc = 1.96 * std(patt_corr_spoc, 0, 2) / sqrt(N); ci_patt_ica = 1.96 * std(patt_corr_ica, 0, 2) / sqrt(N);

% --- Настройка оси X ---
snr_powers = log10(snr_vec); 

% 1. Берем каждый второй индекс
idx_ticks = 1:2:length(snr_powers);

% 2. Находим индекс, максимально близкий к 0 (10^0)
[~, zero_idx] = min(abs(snr_powers - 0));

% 3. Объединяем индексы и сортируем их (чтобы 0 встал на свое место в ряду)
final_tick_indices = unique(sort([idx_ticks, zero_idx]));
ticks_to_show = snr_powers(final_tick_indices);

% 4. Генерируем подписи. Если значение 0, пишем '10^{0}', если нет - стандартно.
x_labels = cell(1, length(ticks_to_show));
for i = 1:length(ticks_to_show)
    if abs(ticks_to_show(i)) < 1e-9 % проверка на ноль
        x_labels{i} = '10^{0}'; 
    else
        x_labels{i} = sprintf('10^{%.1f}', ticks_to_show(i));
    end
end

% Создаем окно
figure('Name', 'SNR Analysis', 'Position', [100, 100, 1000, 450]);

% Список данных для цикличной отрисовки (для компактности кода)
titles = {'Envelope Correlation', 'Spatial Pattern Correlation'};
means_spoc = {mean_env_spoc, mean_patt_spoc};
cis_spoc = {ci_env_spoc, ci_patt_spoc};
means_ica = {mean_env_ica, mean_patt_ica};
cis_ica = {ci_env_ica, ci_patt_ica};

for p = 1:2
    subplot(1, 2, p); hold on;
    
    % SPOC
    h1 = plot(snr_powers, means_spoc{p}, '-o', 'LineWidth', 1.5, 'Color', [0 0.447 0.741]);
    plot(snr_powers, means_spoc{p} + cis_spoc{p}, '--', 'Color', [0 0.447 0.741], 'HandleVisibility', 'off');
    plot(snr_powers, means_spoc{p} - cis_spoc{p}, '--', 'Color', [0 0.447 0.741], 'HandleVisibility', 'off');
    
    % ICA
    h2 = plot(snr_powers, means_ica{p}, '-s', 'LineWidth', 1.5, 'Color', [0.85 0.325 0.098]);
    plot(snr_powers, means_ica{p} + cis_ica{p}, '--', 'Color', [0.85 0.325 0.098], 'HandleVisibility', 'off');
    plot(snr_powers, means_ica{p} - cis_ica{p}, '--', 'Color', [0.85 0.325 0.098], 'HandleVisibility', 'off');
    
    % Оформление
    grid on;
    title(titles{p});
    xlabel('SNR (as Envelope Variance)');
    ylabel('Correlation');
    xticks(ticks_to_show);
    xticklabels(x_labels);
    xlim([min(snr_powers) max(snr_powers)]);
    if p == 1, legend([h1, h2], {'Envelope-CorrCA', 'ICA'}, 'Location', 'southeast'); end
end

%%