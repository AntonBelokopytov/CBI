close all
clear
clc

ft_path = 'C:\Users\ansbel\Documents\2Git\fieldtrip\fieldtrip';

if ~exist('ft_defaults','file')
    addpath(ft_path);
end

%% Target epochs
data = load('epochs_dataset.mat')

x = data.ch_pos(1:55,1);
y = data.ch_pos(1:55,2);
z = data.ch_pos(1:55,3);

%%
% X = data.epochs;
% X = X(:,1:55,:);
% 
% Fmin = 30;
% Fmax = 70;
% band = [Fmin Fmax];
% [b_band,a_band] = butter(3, band/(Fs/2));
% [b_notch, a_notch] = butter(4, [45 55]/(Fs/2), 'stop');
% 
% Xraw = []; 
% for ep_idx=1:size(X,1)
%     ep_idx
%     Ep = squeeze(X(ep_idx,:,:))';
%     Ep_notch = filtfilt(b_notch, a_notch, Ep);
%     Epfilt = filtfilt(b_band, a_band, Ep_notch);
% 
%     Xraw = [Xraw;Epfilt(Fs+1:end-Fs,:)];
% end
% 
% %%
% [U,S,~] = svd(Xraw','econ');
% S = diag(S);
% 
% % Estimate effective rank
% clear eps
% tol = max(size(Xraw)) * eps(S(1));
% r = sum(S > tol);
% 
% % Cumulative variance explained
% ve = S.^2;
% var_explained = cumsum(ve) / sum(ve);
% var_explained(end) = 1;
% 
% % Number of components explaining at least 99% variance
% n_components = find(var_explained>=0.99, 1);
% n_components = max(min(n_components, r), 1);
% n_components
% 
% U = U(:,1:n_components);               % Keep relevant PCA components
% 
% %%
% X = data.epochs;
% X = X(:,1:55,:);
% 
% Fs = data.sfreq;
% Ws = 2;
% Ss = 0.5;
% 
% Epochs = []; Trials_epochs = []; marks = [];
% for ep_idx=1:size(X,1)
%     ep_idx
%     Ep = squeeze(X(ep_idx,:,:))';
%     Ep_notch = filtfilt(b_notch, a_notch, Ep);
%     Epfilt = filtfilt(b_band, a_band, Ep_notch);
% 
%     Epochs(:,:,ep_idx) = Epfilt(Fs+1:end-Fs,:) * U;
% 
%     epes = epoch_data(Epfilt(Fs+1:end-Fs,:)* U,Fs,Ws,Ss);
%     Trials_epochs = cat(3, Trials_epochs, epes);
% 
%     n = data.events(ep_idx,3);
%     marks = [marks, repmat(n, 1, size(epes, 3))];
% end
% 
% Epochs_cov = [];
% for ep_idx=1:size(Trials_epochs,3)
%     ep_idx
%     Epochs_cov(:,:,ep_idx) = cov(Trials_epochs(:,:,ep_idx));
% end
% 
% Tvecs = Tangent_space(Epochs_cov);
% 
% %%
% Tvecs_ga = Tvecs;
% 
% %%
% Tvecs_all = [Tvecs_al; Tvecs_be; Tvecs_ga];
% size(Tvecs_all)
% 
% %%
% clear u
% u = UMAP("n_neighbors",20,"n_components",3,"min_dist",0);
% u.metric = 'euclidean';
% u.target_metric = 'euclidean';
% R = u.fit_transform(Tvecs');
% 
% %%
% labels = {'rest', 'index', 'middle', 'ring', 'pinky'};
% unique_marks = unique(marks);
% hold on;
% for i = 1:length(unique_marks)
%     idx = (marks == unique_marks(i));
%     scatter3(R(idx,1), R(idx,2), R(idx,3), 36, 'filled', 'DisplayName', labels{i});
% end
% legend('show');
% grid on; view(3);

%%
%%
%%
%%
%%
X = data.epochs;
X = X(:,1:55,:);

Fs = data.sfreq;
Fmin = 15;
Fmax = 25;
band = [Fmin Fmax];
[b_band,a_band] = butter(3, band/(Fs/2));
[b_notch, a_notch] = butter(4, [45 55]/(Fs/2), 'stop');

Epochs = []; Trials_epochs = []; marks = [];
for ep_idx=1:size(X,1)
    ep_idx
    Ep = squeeze(X(ep_idx,:,:))';
    Ep_notch = filtfilt(b_notch, a_notch, Ep);
    Epfilt = filtfilt(b_band, a_band, Ep_notch);
    
    Epochs(:,:,ep_idx) = Epfilt(Fs+1:end-Fs,:);
end

%
mask = data.events(:,3) == 1;
% Epochs_rest = Epochs(end-Fs*1+1:end,:,mask);
Epochs_rest = Epochs(:,:,mask);

mask = data.events(:,3) == 2;
% Epochs_mov = Epochs(1:Fs*2,:,mask);
Epochs_mov = Epochs(:,:,mask);

Epochs_comb = [];
for i=1:10
    Epochs_comb(:,:,i) = cat(1, squeeze(Epochs_rest(:,:,i)), squeeze(Epochs_mov(:,:,i)));
end
% Epochs_comb = Epochs_rest;

% mask = data.events(:,3) ~= 1;
% Epochs_comb = Epochs(:,:,mask);

%%
Wsize = 0.5;
Ssize = 0.1;
method = 'full';

% Epochs_comb = Epochs(1:Fs*2,:,mask);
[W, A, corrs, Epochs_cov, S] = env_corrca(Epochs_comb, Fs, Wsize, Ssize, method);

%%
mask = data.events(:,3) == 1;
Epochs_rest = Epochs(end-Fs*2+1:end,:,mask);
% Epochs_rest = Epochs(:,:,mask);

mask = data.events(:,3) == 2;
Epochs_mov = Epochs(1:Fs*2,:,mask);
% Epochs_mov = Epochs(:,:,mask);

% --- НАЧАЛО БЛОКА АУГМЕНТАЦИИ ---
n_mov_trials = size(Epochs_mov, 3);
n_rest_trials = size(Epochs_rest, 3);
n_aug = 10; % Сколько раз использовать каждый трайл движения

% Предварительно выделяем память (хорошая практика в MATLAB для скорости)
len_rest = size(Epochs_rest, 1);
len_mov = size(Epochs_mov, 1);
n_channels = size(Epochs_mov, 2);
n_total_trials = n_mov_trials * n_aug;

Epochs_comb = zeros(len_rest + len_mov, n_channels, n_total_trials);

counter = 1;
for i = 1:n_mov_trials
    % Генерируем 10 случайных индексов эпох покоя (с возвращением)
    % Если трайлов покоя больше 10 и ты хочешь строго уникальные пары для одного движения, 
    % то замени randi на: rand_rest_idx = randperm(n_rest_trials, n_aug);
    rand_rest_idx = randi(n_rest_trials, 1, n_aug);
    
    for j = 1:n_aug
        rest_idx = rand_rest_idx(j);
        
        % Склеиваем
        comb_ep = cat(1, squeeze(Epochs_rest(:,:,rest_idx)), squeeze(Epochs_mov(:,:,i)));
        
        % Записываем в общий тензор
        Epochs_comb(:,:,counter) = comb_ep;
        counter = counter + 1;
    end
end

%%
Wsize = 0.5;
Ssize = 0.1;
method = 'full';
[W, A, corrs, Epochs_cov, S] = env_corrca(Epochs_comb, Fs, Wsize, Ssize, method);

%% Инициализация и расчеты
com_idx = 55;
w = W(:, com_idx);
a = A(:, com_idx);

% Делаем максимальное абсолютное значение паттерна положительным
[~, imax] = max(abs(a));
sign_flip = sign(a(imax));
a = a * sign_flip;

% В твоем коде матрица Epochs имеет размер [n_times, n_channels, n_epochs]
[n_times, n_channels, n_epochs] = size(Epochs_comb);

% Переставляем размерности в [n_channels, n_times, n_epochs], 
% чтобы каналы шли первыми (нужно для матричного умножения)
Epochs_perm = permute(Epochs_comb, [2, 1, 3]);

% Разворачиваем в 2D матрицу [n_channels, n_times * n_epochs]
Epochs_reshaped = reshape(Epochs_perm, n_channels, []);

% Применяем пространственный фильтр w
Xcomp_reshaped = w' * Epochs_reshaped;

% Сворачиваем обратно, получая матрицу активности компоненты [n_times, n_epochs]
Xcomp = reshape(Xcomp_reshaped, n_times, n_epochs);

% Расчет огибающей
% Функция hilbert работает по столбцам (то есть по времени, что нам идеально подходит)
analytic = hilbert(Xcomp); 
envelope = abs(analytic);

% Создаем кастомную красно-бело-синюю палитру (аналог RdBu_r)
% Синий -> Белый -> Красный
rdbu_cmap = [linspace(0,1,128)', linspace(0,1,128)', ones(128,1);
             ones(128,1), linspace(1,0,128)', linspace(1,0,128)'];

% Для симметрии цветовой шкалы найдем максимальные по модулю значения
vmax_w = max(abs(w));
vmax_a = max(abs(a));

% ==========================================================
% ФИГУРА 1: 3D Scatter (Filter & Pattern)
% ==========================================================
fig1 = figure('Color', 'w', 'Position', [100, 100, 1000, 400]);

% -------- Filter --------
subplot(1, 2, 1);
% scatter3(X, Y, Z, S, C, ...)
scatter3(x, y, z, 80, w, 'filled', 'MarkerEdgeColor', 'k');
colormap(gca, rdbu_cmap);
title('Filter', 'FontSize', 14);
colorbar;
axis off;
view(45, 30); % azim=45, elev=30

% -------- Pattern --------
subplot(1, 2, 2);
scatter3(x, y, z, 80, a, 'filled', 'MarkerEdgeColor', 'k');
colormap(gca, rdbu_cmap);
title('Pattern', 'FontSize', 14);
colorbar;
axis off;
view(45, 30);

% ==========================================================
% ФИГУРА 2: Огибающие и динамика
% ==========================================================
time_sec = (0:(n_times-1)) / Fs;

mean_envelope = mean(envelope, 2)';
std_envelope  = std(envelope, 0, 2)';

fig2 = figure('Color', 'w');
% Используем tiledlayout (доступно с R2019b) для аналога GridSpec
t = tiledlayout(4, 1, 'TileSpacing', 'compact', 'Padding', 'compact');

% 1 Heatmap (занимает 2 плитки по высоте)
ax1 = nexttile([2 1]);
% imagesc отображает ось Y сверху вниз, поэтому задаем 'YDir' = 'normal'
imagesc(time_sec, 1:n_epochs, envelope');
set(gca, 'YDir', 'normal');
colormap(ax1, 'parula'); % аналог viridis
colorbar;
ylabel('Epoch');
title('Envelope per epoch');

% 2 All epochs (raw)
ax2 = nexttile;
hold on;
% Рисуем все линии сразу, используя транспонированную матрицу Xcomp
% Добавляем прозрачность (alpha) через свойство цвета (четвертый элемент RGBA)
plot(time_sec, Xcomp', 'Color', [0 0.4470 0.7410 0.4]); 
ylabel('Amplitude');
title('Component activity (all epochs)');
grid on;
hold off;

% 3 Mean envelope ± SD
ax3 = nexttile;
hold on;
% Создаем полигон для закраски стандартного отклонения (fill_between)
X_fill = [time_sec, fliplr(time_sec)];
Y_fill = [mean_envelope - std_envelope, fliplr(mean_envelope + std_envelope)];
fill(X_fill, Y_fill, [0 0.4470 0.7410], 'FaceAlpha', 0.2, 'EdgeColor', 'none');

% Линия среднего
plot(time_sec, mean_envelope, 'LineWidth', 2, 'Color', [0 0.4470 0.7410]);
ylabel('Envelope');
xlabel('Time (s)');
title('Mean envelope \pm SD');
grid on;
hold off;

% Синхронизация осей X (как sharex=ax3 в питоне)
linkaxes([ax1, ax2, ax3], 'x');
xlim([time_sec(1), time_sec(end)]);
