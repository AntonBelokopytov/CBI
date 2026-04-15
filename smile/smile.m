close all
clear
clc

% Запускаем параллельный пул, если он еще не запущен
poolobj = gcp('nocreate');
if isempty(poolobj)
    parpool;
end

% Настройка FieldTrip
ft_path = 'C:\Users\ansbel\Documents\GitHub\CBI\site-packages\fieldtrip\';
if ~exist('ft_defaults','file')
    addpath(ft_path);
end
ft_defaults; % Инициализация дефолтных настроек FieldTrip (рекомендуется)

% Базовые директории
data_dir = 'D:\OS(CURRENT)\data\smile_lobe\cleaned_data\';
save_dir = 'D:\OS(CURRENT)\data\smile_lobe\embeddings\'; % Папка для сохранения результатов

% Создаем папку для сохранений, если её нет
if ~exist(save_dir, 'dir')
    mkdir(save_dir);
end

% Параметры для анализа
fc_list = 5:1:30; 
num_bands = length(fc_list);
lambda = 0.05; 

%% Главный цикл по испытуемым (1-12)
for sub_idx = 1:12
    % Формируем имя файла и полный путь к нему
    % %03d превратит 1 в '001', 12 в '012' и т.д.
    sub_name = sprintf('E%03d', sub_idx);
    file_name = sprintf('%s_ptp150_ica_interpolated_raw.fif', sub_name);
    sub_path = fullfile(data_dir, file_name);
    
    fprintf('==================================================\n');
    fprintf('Начало обработки испытуемого: %s\n', sub_name);
    fprintf('==================================================\n');
    
    % Проверка существования файла
    if ~exist(sub_path, 'file')
        warning('Файл не найден: %s. Пропуск...', sub_path);
        continue;
    end
    
    % Загрузка данных
    cfg = [];
    cfg.dataset = sub_path; 
    Epochs_inf = ft_preprocessing(cfg); 
    Fs = Epochs_inf.hdr.Fs;
    
    X = Epochs_inf.trial{1};
    n_channels = size(X, 1);
    I = eye(n_channels); 
    
    % Инициализация переменных для хранения результатов текущего испытуемого
    UMAP_results = cell(1, num_bands);
    Epochs_counts = zeros(1, num_bands); 
    
    % Цикл анализа по частотам
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
        
        % Расчет ковариационных матриц
        parfor i = 1:n_epochs
            C_sample = cov(X_epochs(:,:,i));    
            nu = trace(C_sample) / n_channels;    
            C_reg = (1 - lambda) * C_sample + lambda * nu * I;    
            X_covs(:,:,i) = C_reg;
        end
        
        Dists = zeros(n_epochs, n_epochs); 
        
        % Расчет римановых расстояний
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
    
    % Сохранение результатов для испытуемого
    save_filename = fullfile(save_dir, sprintf('%s_umap_embeddings.mat', sub_name));
    save(save_filename, 'UMAP_results', 'Epochs_counts', 'fc_list', 'Fs', 'sub_name');
    fprintf('Вложения для %s успешно сохранены в файл:\n%s\n\n', sub_name, save_filename);
    
    % Очищаем тяжелые переменные перед следующим испытуемым
    clear Epochs_inf X X_filt X_epochs X_covs Dists;
end

fprintf('Обработка всех 12 испытуемых завершена!\n');

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
