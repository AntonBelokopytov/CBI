close all
clear
clc

ft_path = 'C:\Users\ansbel\Documents\2Git\fieldtrip';
if ~exist('ft_defaults','file')
    addpath(ft_path);
end
ft_defaults;

%% ЗАГРУЗКА ДАННЫХ
sub_path = 'ica_raw.fif';

cfg = [];
cfg.dataset = sub_path;
% Выбираем ТОЛЬКО планарные градиометры (суффиксы 2 и 3 в Neuromag)
cfg.channel = 'megplanar'; 
Xinf = ft_preprocessing(cfg);

Fs = Xinf.fsample;

%% НАСТРОЙКА ТОПОГРАФИИ
topo = [];
topo.dimord = 'chan_time';
topo.label  = Xinf.label;  
topo.time   = 0;
topo.grad   = Xinf.grad; % ВАЖНО: используем .grad вместо .elec для МЭГ!

%% ПОДГОТОВКА ЛЕЙАУТА (LAYOUT)
laycfg = [];
% Для Neuromag данных лучше всего использовать стандартный шаблон FieldTrip
% Это даст самую красивую и правильную расстановку сенсоров.
laycfg.layout = 'neuromag306planar.lay'; 

% (Альтернативный вариант: сгенерировать из данных)
% laycfg.grad = Xinf.grad; 

lay = ft_prepare_layout(laycfg);     

%% ПАРАМЕТРЫ ДЛЯ ОТРИСОВКИ
% Рекомендую создавать новый cfg для отрисовки, чтобы настройки препроцессинга не мешали
cfg_plot = []; 
cfg_plot.marker       = 'labels';
cfg_plot.layout       = lay;
cfg_plot.comment      = 'no';
cfg_plot.style        = 'fill';
cfg_plot.markersymbol = 'o';
cfg_plot.colorbar     = 'no';

%% ПОДГОТОВКА ДАННЫХ ДЛЯ ОТРИСОВКИ
% Создаем структуру, понятную для ft_topoplotER
topo = [];
topo.dimord = 'chan_time';
topo.label  = Xinf.label;  
topo.time   = 0;
topo.grad   = Xinf.grad; 

% Считаем дисперсию (размах активности) по времени для каждого из 204 каналов
% Xinf.trial{1} имеет размер [204 канала x 901250 отсчетов]
topo.avg = var(Xinf.trial{1}, 0, 2); 

%% ОТРИСОВКА ТОПОГРАФИИ
cfg_plot = []; 
cfg_plot.layout       = lay;      % Тот самый layout 'neuromag306planar.lay'
cfg_plot.marker       = 'labels'; % Показываем названия (например, MEG0112)
cfg_plot.comment      = 'no';
cfg_plot.style        = 'fill';
cfg_plot.markersymbol = '.';      % Точки вместо кружочков смотрятся аккуратнее
cfg_plot.colorbar     = 'yes';    % Включаем шкалу, чтобы понимать масштаб значений

figure('Name', 'MEG Planar Gradiometers Topography', 'Color', 'w');
ft_topoplotER(cfg_plot, topo);
title('Активность планарных градиометров (дисперсия сигнала)');

%% ФИЛЬТРАЦИЯ ДАННЫХ (Base MATLAB way)
[b,a] = butter(3, [30,45]/(Fs/2), 'bandpass'); 

% Создаем копию структуры, чтобы не испортить оригинал
Xinf_filtered_matlab = Xinf;

% Вытаскиваем матрицу данных (204 канала х 901250 отсчетов)
% Обязательно переводим в double, иначе filtfilt может выдать ошибку
rawData = double(Xinf.trial{1});

% Применяем filtfilt. 
filteredData = filtfilt(b, a, rawData')';

% Возвращаем отфильтрованные данные в структуру
Xinf_filtered_matlab.trial{1} = filteredData;
X = Xinf_filtered_matlab.trial{1};

%%
[U,S,~] = svd(X,'econ');
S = diag(S);
ve = S.^2;
var_explained = cumsum(ve) / sum(ve);
var_explained(end) = 1; % Защита от ошибок округления
tol = 10e-6;
n_components = find(var_explained >= 1 - tol, 1);
U = U(:,1:n_components);
Xpca = U' * X; % Размерность: [n_components x время]
n_components

%%
Ws = 5;
Ss = 1;

X_epo = epoch_data(Xpca',Fs,Ws,Ss);
time = 1:size(X,2);
time_epochs = epoch_data(time',Fs,Ws,Ss);

Covs = [];
for ep_idx=1:size(X_epo,3)
    Covs(:,:,ep_idx) = cov(X_epo(:,:,ep_idx));
end

%%
Tcovs = Tangent_space(Covs);

%%
[Ut,St,~] = svd(Tcovs,'econ');
St = diag(St);
ve = St.^2;
var_explained = cumsum(ve) / sum(ve);
var_explained(end) = 1; % Защита от ошибок округления
tol = 10e-6;
n_components = find(var_explained >= 1 - tol, 1);
Ut = Ut(:,1:n_components);
Tcovspca = Ut' * Tcovs; % Размерность: [n_components x время]
Tcovspca = Tcovspca';

%%
plot(Tcovspca(:,4:7))

%%
clear u
u = UMAP('n_neighbors', 20, 'n_components', 3, 'metric', 'euclidean');

R = u.fit_transform(Tcovs');

%%
scatter3(R(:,1),R(:,2),R(:,3))

%% БЛОК: Разметка состояний и визуализация
t1 = [0, 450];
t2 = [450, 1160];
t3 = [1160, 2295];
t4 = [2295, 3185];
t5 = [3185, 3600];
% 1. Оформляем интервалы в матрицу (через точку с запятой) и имена в cell array
times = [t1; t2; t3; t4; t5]; 
names = {'1A', '2S', '3A', '4S', '5A'};

% 2. Размечаем эпохи
num_epochs = size(R, 1);
labels = zeros(num_epochs, 1);

for i = 1:num_epochs
    % Берем средний отсчет времени для текущей эпохи и переводим в секунды
    % (предполагается, что time_epochs имеет размер [отсчеты x 1 x эпохи])
    mid_sample = mean(time_epochs(:, i)); 
    t_sec = mid_sample / Fs; 
    
    % Проверяем, в какой интервал попали эти секунды
    for j = 1:size(times, 1)
        if t_sec >= times(j, 1) && t_sec < times(j, 2)
            labels(i) = j;
            break;
        end
    end
end

% 3. Отрисовка 3D графика с легендой
figure('Name', 'UMAP MEG States', 'Color', 'w');
hold on; grid on;

% Зададим контрастные цвета для разных состояний
colors = [
    0.85, 0.15, 0.20;  % 1_awake (красный)
    0.15, 0.35, 0.80;  % 2_asleep (синий)
    0.95, 0.50, 0.10;  % 3_awake (оранжевый)
    0.30, 0.70, 0.95;  % 4_asleep (голубой)
    0.90, 0.75, 0.15   % 5_awake (желтый)
];

% Рисуем точки группами, чтобы легенда собралась правильно
for j = 1:5
    idx = (labels == j); % Находим все эпохи j-го состояния
    
    if any(idx)
        scatter3(R(idx, 1), R(idx, 2), R(idx, 3), ...
                 25, colors(j,:), 'filled', ...
                 'MarkerFaceAlpha', 0.8, ...
                 'DisplayName', names{j});
    end
end

title('UMAP Проекция состояний мозга');
xlabel('UMAP 1');
ylabel('UMAP 2');
zlabel('UMAP 3');
legend('show', 'Location', 'best');
view(-45, 30); 
