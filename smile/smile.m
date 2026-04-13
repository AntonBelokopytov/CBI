close all
clear
clc

% Запускаем параллельный пул, если он еще не запущен
% poolobj = gcp('nocreate');
% if isempty(poolobj)
%     parpool;
% end

ft_path = 'C:\Users\anton\Documents\GitHub\CBI\site-packages\fieldtrip\';
if ~exist('ft_defaults','file')
    addpath(ft_path);
end

%% Загрузка данных
sub_path = 'D:\OS(CURRENT)\data\smile_lobe\cleaned_data\E001_ptp150_ica_interpolated_raw.fif';
cfg = [];
cfg.dataset = sub_path; 
Epochs_inf = ft_preprocessing(cfg); 
Fs = Epochs_inf.hdr.Fs;

%%
X = Epochs_inf.trial{1};
n_channels = size(X, 1);

%% Цикл анализа по частотам
fc_list = 25:2:40; 
num_bands = length(fc_list);

UMAP_results = cell(1, num_bands);
Epochs_counts = zeros(1, num_bands); 

lambda = 0.05; 
I = eye(n_channels); 

for fb = 1:num_bands
    Fc = fc_list(fb);
    
    band_halfwidth = max(2, Fc * 0.20);
    Fmin = Fc - band_halfwidth;
    Fmax = Fc + band_halfwidth;
    band = [Fmin, Fmax];
    
    % Фильтрация
    [b_band, a_band] = butter(2, band/(Fs/2));
    X_filt = filtfilt(b_band, a_band, X')';
    
    Ws = 1/Fc; 
    Ss = Ws;
    X_epochs = epoch_data(X_filt', Fs, Ws, Ss);
    
    n_epochs = size(X_epochs, 3);
    Epochs_counts(fb) = n_epochs;
    fprintf('Частота %d Гц: окно %.3f с, эпох: %d\n', Fc, Ws, n_epochs);
    
    X_covs = zeros(n_channels, n_channels, n_epochs);
    parfor i = 1:n_epochs
        C_sample = cov(X_epochs(:,:,i));    
        nu = trace(C_sample) / n_channels;    
        C_reg = (1 - lambda) * C_sample + lambda * nu * I;    
        X_covs(:,:,i) = C_reg;
    end
    
    Dists = zeros(n_epochs, n_epochs); 
    
    parfor i = 1:n_epochs-1
        row_dists = zeros(1, n_epochs);
        C1 = X_covs(:,:,i);        
        C1_inv = inv(C1); 
        
        for j = i+1:n_epochs
            C2 = X_covs(:,:,j);
            
            eigs_val = eig(C1_inv * C2);            
            row_dists(j) = sqrt(sum(log(eigs_val).^2));
        end
        Dists(i, :) = row_dists;     
    end
    
    Dists = Dists + Dists';
    Dists(logical(eye(n_epochs))) = 0; 
    
    % --- UMAP ---
    fprintf('Считаем UMAP для %d Гц...\n', Fc);
    u = UMAP('n_neighbors', 20, 'n_components', 2, 'metric', 'precomputed');
    UMAP_results{fb} = u.fit_transform(Dists); 
    fprintf('--- Готово для %d Гц ---\n\n', Fc);
end

%% Визуализация всех вложений на одном рисунке
figure('Name', 'Риманово многообразие по частотам', 'Color', 'w', 'Position', [50, 50, 1600, 900]);

cols = ceil(sqrt(num_bands));
rows = ceil(num_bands / cols);

for fb = 1:num_bands
    subplot(rows, cols, fb);
    
    R = UMAP_results{fb};
    
    scatter(R(:,1), R(:,2), 10, 'filled', 'MarkerFaceAlpha', 0.6);
    title(sprintf('Fc = %d Гц (Эпох: %d)', fc_list(fb), Epochs_counts(fb)), 'FontSize', 10);
    
    xticks([]); yticks([]); 
    axis square;
end