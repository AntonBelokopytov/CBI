close all
clear
clc

% Запускаем пул потоков для максимальной скорости (если не запущен)
poolobj = gcp('nocreate');
if isempty(poolobj)
    parpool('Threads'); 
end

% --- Настройка путей ---
ft_path = 'C:\Users\anton\Documents\GitHub\CBI\site-packages\fieldtrip\';
if ~exist('ft_defaults','file')
    addpath(ft_path);
end
ft_defaults;

data_dir = 'D:\OS(CURRENT)\data\parkinson\control\';
idx_dir  = 'D:\OS(CURRENT)\data\parkinson\indices_simple\'; % Папка, куда сохраняли .mat из Python
out_dir  = 'D:\OS(CURRENT)\data\parkinson\figures\'; % Папка для сохранения графиков

if ~exist(out_dir, 'dir')
    mkdir(out_dir);
end

% Находим все файлы контрольной группы
fif_files = dir(fullfile(data_dir, 'Control_*_CenterOut_epochs.fif'));
num_subjects = length(fif_files);
fprintf('Найдено файлов для обработки: %d\n', num_subjects);

%% Главный цикл по испытуемым
for sub = 1:num_subjects
    % Получаем имя файла и формируем пути
    fif_filename = fif_files(sub).name;
    sub_path = fullfile(data_dir, fif_filename);
    
    % Извлекаем базовое имя (например, 'Control_1')
    sub_name = strrep(fif_filename, '_CenterOut_epochs.fif', '');
    
    % Формируем путь к файлу с индексами
    idx_path = fullfile(idx_dir, sprintf('%s_idx.mat', sub_name));
    
    fprintf('==================================================\n');
    fprintf('Начало обработки: %s\n', sub_name);
    
    % Проверяем наличие файла с индексами
    if ~exist(idx_path, 'file')
        warning('Файл индексов не найден: %s. Пропуск испытуемого...', idx_path);
        continue;
    end
    
    % Загружаем индексы (переменная 'idx', сохраненная из Python)
    loaded_data = load(idx_path);
    idxs = loaded_data.idx;
    
    if isempty(idxs)
        warning('Список индексов пуст для %s. Пропуск...', sub_name);
        continue;
    end
    
    % Загрузка данных ЭЭГ
    cfg = [];
    cfg.dataset = sub_path; 
    Epochs_inf = ft_preprocessing(cfg); 
    Fs = Epochs_inf.hdr.Fs;
    n_trials = numel(Epochs_inf.trial);
    
    % Извлекаем ячейки с эпохами в отдельную переменную ДО parfor
    % Это ускорит передачу данных (broadcast) в потоки
    trials_cell = Epochs_inf.trial;
    valid_idxs = idxs(idxs <= n_trials); 
    
    % Настройки частот
    fc_list = 5:30;          
    num_bands = length(fc_list);
    num_comps_to_analyze = 10; 
    results_corr = zeros(num_bands, num_comps_to_analyze);
    results_var  = zeros(num_bands, num_comps_to_analyze);
    
    fprintf('Запуск параллельного анализа частот...\n');
    
    % --- ПАРАЛЛЕЛЬНЫЙ ЦИКЛ ПО ЧАСТОТАМ ---
    parfor fb = 1:num_bands
        Fc = fc_list(fb);
        
        band_halfwidth = max(2, Fc * 0.10);
        Fmin = Fc - band_halfwidth;
        Fmax = Fc + band_halfwidth;
        band = [Fmin Fmax];
        
        Wsize = 1/Fc; 
        Ssize = Wsize/2;
        
        [b_band, a_band] = butter(2, band/(Fs/2)); 
        
        % Временные переменные для хранения результатов текущей частоты
        % (Требование parfor для корректного slicing-а матриц)
        temp_corr = zeros(1, num_comps_to_analyze);
        temp_var  = zeros(1, num_comps_to_analyze);
        
        % Преаллокация для скорости (узнаем размеры по 1-й эпохе)
        Ep_test = trials_cell{1}';
        Ep_test = Ep_test(:, 1:38);
        Epfilt_test = filtfilt(b_band, a_band, Ep_test);
        alg_test = Epfilt_test(Fs/2+1:end-Fs/2,:);
        
        % Локальные массивы внутри потока
        local_Epochs = zeros(size(Epfilt_test, 1), size(Epfilt_test, 2), n_trials);
        local_Epochs_alg = zeros(size(alg_test, 1), size(alg_test, 2), n_trials);
        
        for ep_idx = 1:n_trials  
            Ep = trials_cell{ep_idx}';
            Ep = Ep(:, 1:38);
            Epfilt = filtfilt(b_band, a_band, Ep);
            local_Epochs(:,:,ep_idx) = Epfilt;
            local_Epochs_alg(:,:,ep_idx) = Epfilt(Fs/2+1:end-Fs/2,:);
        end
        
        % Оставляем только нужные эпохи по индексам
        local_Epochs = local_Epochs(:,:,valid_idxs);
        local_Epochs_alg = local_Epochs_alg(:,:,valid_idxs);
        
        % Вызов вашей функции
        [~, ~, z_trials, ~, raw_var] = env_corrca(local_Epochs_alg, Fs, Wsize, Ssize);
        
        [~, total_comps, ~] = size(z_trials);
        comps_limit = min(total_comps, num_comps_to_analyze);
        
        for c = 1:comps_limit
            comp_data = squeeze(z_trials(:, c, :)); 
            R = corr(comp_data); 
            upper_tri_idx = triu(true(size(R)), 1); 
            
            temp_corr(c) = mean(R(upper_tri_idx));
            temp_var(c)  = mean(raw_var(c, :));
        end
        
        % Записываем локальные результаты в общую матрицу
        results_corr(fb, :) = temp_corr;
        results_var(fb, :)  = temp_var;
        
    end % Конец parfor
    
    fprintf('Расчеты завершены для %s. Сохранение графиков...\n', sub_name);
    
    % Визуализация и сохранение
    x_values = fc_list; 
    colors = lines(num_comps_to_analyze); 
    
    % --- График 1: Межтрайловая корреляция (ITC) ---
    fig_corr = figure('Name', 'Inter-trial Correlation', 'Color', 'w', ...
                      'Position', [100, 100, 800, 500], 'Visible', 'off');
    hold on; grid on;
    for c = 1:num_comps_to_analyze
        % Защита, если компонент меньше, чем num_comps_to_analyze
        if any(results_corr(:, c)) 
            plot(x_values, results_corr(:, c), '-o', 'LineWidth', 2, ...
                'Color', colors(c,:), 'MarkerSize', 6, 'MarkerFaceColor', colors(c,:), ...
                'DisplayName', sprintf('Comp %d', c));
        end
    end
    xticks(x_values); 
    xlim([min(x_values)-1, max(x_values)+1]);
    xlabel('Центральная частота, Гц', 'FontSize', 12, 'FontWeight', 'bold');
    ylabel('Коэффициент корреляции (ITC)', 'FontSize', 12, 'FontWeight', 'bold');
    title(sprintf('Спектр межтрайловой корреляции компонент - %s', strrep(sub_name, '_', '\_')), 'FontSize', 14);
    legend('show', 'Location', 'bestoutside'); 
    set(gca, 'FontSize', 11);
    
    saveas(fig_corr, fullfile(out_dir, sprintf('%s_ITC.png', sub_name)));
    close(fig_corr); 
    
    % --- График 2: Средняя дисперсия ---
    fig_var = figure('Name', 'Average Variance', 'Color', 'w', ...
                     'Position', [150, 150, 800, 500], 'Visible', 'off');
    hold on; grid on;
    for c = 1:num_comps_to_analyze
        if any(results_var(:, c))
            plot(x_values, results_var(:, c), '-s', 'LineWidth', 2, ...
                'Color', colors(c,:), 'MarkerSize', 6, 'MarkerFaceColor', colors(c,:), ...
                'DisplayName', sprintf('Comp %d', c));
        end
    end
    xticks(x_values);
    xlim([min(x_values)-1, max(x_values)+1]);
    xlabel('Центральная частота, Гц', 'FontSize', 12, 'FontWeight', 'bold');
    ylabel('Средняя дисперсия', 'FontSize', 12, 'FontWeight', 'bold');
    title(sprintf('Спектр дисперсии компонент - %s', strrep(sub_name, '_', '\_')), 'FontSize', 14);
    legend('show', 'Location', 'bestoutside');
    set(gca, 'FontSize', 11);
    
    saveas(fig_var, fullfile(out_dir, sprintf('%s_Variance.png', sub_name)));
    close(fig_var);
    
    % Очистка тяжелых переменных перед следующим испытуемым
    clear Epochs_inf trials_cell;
end

fprintf('Ура! Все данные обработаны и графики сохранены.\n');
