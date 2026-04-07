close all
clear
clc

ft_path = 'C:\Users\ansbel\Documents\GitHub\CBI\site-packages\fieldtrip\';

if ~exist('ft_defaults','file')
    addpath(ft_path);
end

%% Target epochs
% sub_path = 'D:\OS(CURRENT)\data\parkinson\pathology\Patient_1_CenterOut_OFF_EEG_clean_epochs.fif';
sub_path = 'D:\OS(CURRENT)\data\parkinson\control\Control_3_CenterOut_epochs.fif';
cfg = [];
cfg.dataset = sub_path; 
Epochs_inf = ft_preprocessing(cfg); 

Fs = Epochs_inf.hdr.Fs;

[~, n_ts_ep] = size(Epochs_inf.trial{1});

%%
% idxs = [  0,   1,   2,   3,   6,   8,  12,  13,  15,  16,  17,  19,  22,...
%         24,  25,  28,  30,  32,  33,  35,  37,  39,  41,  42,  44,  45,...
%         48,  50,  51,  55,  57,  58,  59,  62,  63,  65,  69,  70,  73,...
%         74,  75,  78,  79,  82,  85,  88,  89,  92,  93,  94,  97, 100,...
%        101, 103, 105, 106, 110, 111, 112, 117, 118, 121, 122, 124, 126,...
%        128, 131, 132, 134, 136, 137, 138] + 1;
idxs = [  1,   3,   4,   5,   7,   8,  12,  14,  15,  17,  21,  22,  25,...
        26,  28,  29,  31,  33,  36,  37,  39,  40,  42,  44,  48,  49,...
        50,  53,  55,  56,  59,  61,  63,  64,  66,  67,  71,  72,  73,...
        74,  76,  77,  80,  81,  85,  86,  89,  90,  92,  94,  98,  99,...
       101, 103, 108, 109, 110, 111, 114, 116, 117, 120, 121, 124, 125,...
       127, 128, 130, 131, 134, 136, 138, 141, 142, 144, 146, 147] + 1;

all_idxd = 1:numel(Epochs_inf.trial);
% idxs = setdiff(all_idxd, idxs); 

% ---------------------------------------------------------
% НОВЫЙ БЛОК: ЗАДАНИЕ ЦЕНТРАЛЬНЫХ ЧАСТОТ
% ---------------------------------------------------------
% ---------------------------------------------------------
% ОПТИМИЗИРОВАННЫЙ БЛОК: ДИНАМИЧЕСКАЯ ШИРИНА ПОЛОСЫ
% ---------------------------------------------------------
fc_list = 5:35;          
num_bands = length(fc_list);
num_comps_to_analyze = 10; 

results_corr = zeros(num_bands, num_comps_to_analyze);
results_var  = zeros(num_bands, num_comps_to_analyze);

Fs = Epochs_inf.hdr.Fs;

for fb = 1:num_bands
    Fc = fc_list(fb);
    
    % Делаем ширину окна зависящей от частоты (например, +/- 15% от Fc)
    % Для 10 Гц это будет +/- 1.5 Гц (7.5 - 11.5 Гц)
    % Для 20 Гц это будет +/- 3.0 Гц (17.0 - 23.0 Гц)
    band_halfwidth = max(2, Fc * 0.20);

    Fmin = Fc - band_halfwidth;
    Fmax = Fc + band_halfwidth;
    band = [Fmin Fmax];
    
    % Для Wsize логичнее использовать центральную частоту
    Wsize = 1/Fc; 
    Ssize = Wsize/2;
    
    % Уменьшаем порядок фильтра до 2 (туда-обратно даст 4), 
    % чтобы избежать нестабильности фильтра на низких частотах
    [b_band, a_band] = butter(2, band/(Fs/2)); 
    
    clear Epochs Epochs_alg
    for ep_idx = 1:numel(Epochs_inf.trial)  
        Ep = Epochs_inf.trial{ep_idx}';
        Ep = Ep(:, 1:38);
        Epfilt = filtfilt(b_band, a_band, Ep);
        Epochs(:,:,ep_idx) = Epfilt;
        Epochs_alg(:,:,ep_idx) = Epfilt(Fs/2+1:end-Fs/2,:);
    end
    
    Epochs = Epochs(:,:,idxs);
    Epochs_alg = Epochs_alg(:,:,idxs);
    
    [W, A, z_trials, X_epochs, raw_var] = env_corrca(Epochs_alg, Fs, Wsize, Ssize);
    
    [~, total_comps, n_trials] = size(z_trials);
    comps_limit = min(total_comps, num_comps_to_analyze);
    
    for c = 1:comps_limit
        comp_data = squeeze(z_trials(:, c, :)); 
        R = corr(comp_data); 
        upper_tri_idx = triu(true(size(R)), 1); 
        
        results_corr(fb, c) = mean(R(upper_tri_idx));
        results_var(fb, c)  = mean(raw_var(c, :));
    end
    
    fprintf('Обработана частота %d Гц (диапазон %.1f - %.1f Гц)\n', Fc, Fmin, Fmax);
end

%%
% ---------------------------------------------------------
% ВИЗУАЛИЗАЦИЯ РЕЗУЛЬТАТОВ
% ---------------------------------------------------------
% Теперь ось X - это просто вектор наших центральных частот
x_values = fc_list; 
colors = lines(comps_limit); 

% --- График 1: Межтрайловая корреляция (ITC) ---
figure('Name', 'Inter-trial Correlation', 'Color', 'w', 'Position', [100, 100, 800, 500]);
hold on; grid on;
for c = 1:comps_limit
    plot(x_values, results_corr(:, c), '-o', 'LineWidth', 2, ...
        'Color', colors(c,:), 'MarkerSize', 6, 'MarkerFaceColor', colors(c,:), ...
        'DisplayName', sprintf('Comp %d', c));
end

% Настраиваем ось X как непрерывную шкалу
xticks(x_values); 
xlim([min(x_values)-1, max(x_values)+1]);
xlabel('Центральная частота, Гц', 'FontSize', 12, 'FontWeight', 'bold');
ylabel('Коэффициент корреляции (ITC)', 'FontSize', 12, 'FontWeight', 'bold');
title('Спектр межтрайловой корреляции компонент', 'FontSize', 14);
legend('show', 'Location', 'bestoutside'); 
set(gca, 'FontSize', 11);

% --- График 2: Средняя дисперсия ---
figure('Name', 'Average Variance', 'Color', 'w', 'Position', [150, 150, 800, 500]);
hold on; grid on;
for c = 1:comps_limit
    plot(x_values, results_var(:, c), '-s', 'LineWidth', 2, ...
        'Color', colors(c,:), 'MarkerSize', 6, 'MarkerFaceColor', colors(c,:), ...
        'DisplayName', sprintf('Comp %d', c));
end

xticks(x_values);
xlim([min(x_values)-1, max(x_values)+1]);
xlabel('Центральная частота, Гц', 'FontSize', 12, 'FontWeight', 'bold');
ylabel('Средняя дисперсия', 'FontSize', 12, 'FontWeight', 'bold');
title('Спектр дисперсии компонент', 'FontSize', 14);
legend('show', 'Location', 'bestoutside');
set(gca, 'FontSize', 11);
