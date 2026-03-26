close all
clear
clc

ft_path = 'C:\Users\ansbel\Documents\2Git\fieldtrip';

if ~exist('ft_defaults','file')
    addpath(ft_path);
end

ft_defaults;

%%
sub_path = 'sound_erp_Tum.fif';

cfg = [];
cfg.dataset = sub_path;
Xinf = ft_preprocessing(cfg);
Fs = Xinf.fsample;

topo = [];
topo.dimord = 'chan_time';
topo.label  = Xinf.elec.label;  
topo.time   = 0;
topo.elec   = Xinf.elec;

laycfg = [];
laycfg.elec = Xinf.elec;
lay = ft_prepare_layout(laycfg);     

cfg.marker       = 'labels';
cfg.layout       = lay;
cfg.comment      = 'no';
cfg.style        = 'fill';
cfg.markersymbol = 'o';
cfg.colorbar     = 'no'; 

%%
X = Xinf.trial{1};

Xraw = []; Xtrials = [];
for i=1:numel(Xinf.trial)
    Xraw = cat(2,Xraw,Xinf.trial{i}(1:38,:));
    Xtrials(:,:,i) = Xinf.trial{i}(1:38,:);
end

size(Xraw)

%%
[L, D, W_n, W] = laplace_embedding(mean(Xtrials,3)',50,50);

%%
% [Ux,Sx,~] = svd(Xraw,'econ');
% Ux = Ux(:,1:35);
% Xrawpca = Ux' * Xraw;
% 
% vLv = Xraw * L * Xraw';
% vDv = Xraw * D * Xraw';

[U,S] = eigs(L,D, 1+10,'smallestreal');
S = diag(S);
stem(S)

U = U(:,1:end);

%%
w = canoncorr(mean(Xtrials,3)', Sources);
% w = Ux * U;
a = cov(Xraw') * w;
topo.avg   = a(:,1);

ft_topoplotER(cfg, topo);

%%
scatter3(Sources(:,1),Sources(:,2),Sources(:,3))

%% Расчет и визуализация ERP для источников
num_trials = numel(Xinf.trial);
num_channels = 38; % как в вашем цикле сборки Xraw
num_timepoints = size(Xinf.trial{1}, 2);
num_components = size(w, 2); % Количество найденных компонент

% 1. Создаем пустой массив для хранения источников всех трайлов
% Размерность: [Компоненты x Время x Трайлы]
source_trials = zeros(num_components, num_timepoints, num_trials);

% 2. Проецируем каждый трайл в пространство источников
for i = 1:num_trials
    trial_data = Xinf.trial{i}(1:num_channels, :); % Данные одного трайла [Ch x Time]
    
    % Умножаем транспонированную матрицу весов на данные трайла
    % w имеет размер [Ch x Comp], trial_data - [Ch x Time]
    % Результат - [Comp x Time]
    source_trials(:, :, i) = w' * trial_data; 
end

% 3. Усредняем по третьему измерению (по трайлам), чтобы получить ERP
source_erp = mean(source_trials, 3); 

% 4. Отрисовка усредненных ERP для первых нескольких компонент
time_vector = Xinf.time{1}; % Достаем вектор времени из структуры FieldTrip для оси X

figure('Name', 'Source ERPs', 'Color', 'w', 'Position', [100 100 800 800]);
plot_comps = min(3, num_components); % Отрисуем первые 5 штук

for k = 1:plot_comps
    subplot(plot_comps, 1, k);
    plot(time_vector, source_erp(k, :), 'LineWidth', 1.5, 'Color', '#0072BD');
    
    title(['Component ' num2str(k) ' ERP']);
    ylabel('Amplitude');
    grid on;
    
    % Подписи оси X только для нижнего графика
    if k == plot_comps
        xlabel('Time (s)');
    end
end

%%