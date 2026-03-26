close all
clear
clc

ft_path = 'C:\Users\ansbel\Documents\2Git\fieldtrip\fieldtrip';

if ~exist('ft_defaults','file')
    addpath(ft_path);
end

%%
init_data = load('data_64_64.mat')

%% Target epochs
data = load('epochs_dataset.mat')

x = data.ch_pos(1:55,1);
y = data.ch_pos(1:55,2);
z = data.ch_pos(1:55,3);

%%
X = data.epochs;
X = X(:,1:55,:);

Fs = data.sfreq;
Fmin = 8;
Fmax = 12;
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

%%
mask = data.events(:,3) == 1;
% Epochs_rest = Epochs(end-Fs*2+1:end,:,mask);
Epochs_rest = Epochs(:,:,mask);

mask = data.events(:,3) == 5;
% Epochs_mov = Epochs(1:Fs*5,:,mask);
Epochs_mov = Epochs(:,:,mask);

Epochs_comb = [];
for i=1:10
    Epochs_comb(:,:,i) = cat(1, squeeze(Epochs_rest(:,:,i)), squeeze(Epochs_mov(:,:,i)));
end
% Epochs_comb = Epochs_mov;

%%
Wsize = 0.5;
Ssize = 0.1;
method = 'full';

% Epochs_comb = Epochs(1:Fs*2,:,mask);
[W, A, corrs, Epochs_cov, S] = env_corrca(Epochs_comb, Fs, Wsize, Ssize, method);

%% Инициализация и расчеты
com_idx = 1;
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
% ФИГУРА 1: 3D Scatter (Filter & Pattern) с осями в центре сетки
% ==========================================================
fig1 = figure('Color', 'w', 'Position', [100, 100, 1000, 400]);

% 1. Находим геометрический центр сетки электродов
cx = (max(x) + min(x)) / 2;
cy = (max(y) + min(y)) / 2;
cz = (max(z) + min(z)) / 2;

% 2. Находим максимальный размах от центра, чтобы оси были одинаковой длины (сохраняем куб)
max_dist = max([max(x)-min(x), max(y)-min(y), max(z)-min(z)]) / 2;
r = max_dist * 1.1; % Добавляем 10% отступа (padding) для красоты

% -------- Filter --------
subplot(1, 2, 1);
scatter3(x, y, z, 80, w, 'filled', 'MarkerEdgeColor', 'k');
colormap(gca, rdbu_cmap);
title('Filter', 'FontSize', 14);
colorbar;

% Включаем оси и делаем масштаб пропорциональным
axis on;
axis equal;
grid on;
hold on;

% 3. Рисуем перекрестие осей через центр сетки (cx, cy, cz)
plot3([cx-r, cx+r], [cy, cy], [cz, cz], 'k--', 'LineWidth', 1, 'Color', [0.5 0.5 0.5]); % Ось X
plot3([cx, cx], [cy-r, cy+r], [cz, cz], 'k--', 'LineWidth', 1, 'Color', [0.5 0.5 0.5]); % Ось Y
plot3([cx, cx], [cy, cy], [cz-r, cz+r], 'k--', 'LineWidth', 1, 'Color', [0.5 0.5 0.5]); % Ось Z

% Подписываем оси (стандарт MNI)
xlabel('X (Right)');
ylabel('Y (Nose)');
zlabel('Z (Up)');

% Устанавливаем угол обзора и границы вокруг нового центра
view(-45, 30); 
xlim([cx-r, cx+r]); ylim([cy-r, cy+r]); zlim([cz-r, cz+r]);
hold off;

% -------- Pattern --------
subplot(1, 2, 2);
scatter3(x, y, z, 80, a, 'filled', 'MarkerEdgeColor', 'k');
colormap(gca, rdbu_cmap);
title('Pattern', 'FontSize', 14);
colorbar;

axis on;
axis equal;
grid on;
hold on;

% Рисуем перекрестие осей через центр сетки (cx, cy, cz)
plot3([cx-r, cx+r], [cy, cy], [cz, cz], 'k--', 'LineWidth', 1, 'Color', [0.5 0.5 0.5]); % Ось X
plot3([cx, cx], [cy-r, cy+r], [cz, cz], 'k--', 'LineWidth', 1, 'Color', [0.5 0.5 0.5]); % Ось Y
plot3([cx, cx], [cy, cy], [cz-r, cz+r], 'k--', 'LineWidth', 1, 'Color', [0.5 0.5 0.5]); % Ось Z

xlabel('X (Right \rightarrow)');
ylabel('Y (Nose \rightarrow)');
zlabel('Z (Up \uparrow)');

view(-45, 30);
xlim([cx-r, cx+r]); ylim([cy-r, cy+r]); zlim([cz-r, cz+r]);
hold off

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
