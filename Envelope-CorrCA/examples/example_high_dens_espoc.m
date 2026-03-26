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
% 1. Подготовка осей времени
eeg_length = length(init_data.state); % 507008
xq = (1:eeg_length)';                 % Целевая ось времени (все сэмплы ЭЭГ)

% Исходные координаты [8391 x 2 x 21]
v = init_data.hand_keypoints.location; 

% Защита от "залипших" кадров (interp1 выдаст ошибку, если в x есть дубликаты)
[x_unique, unique_idx] = unique(init_data.hand_keypoints.eeg_sample_num);
v_unique = v(unique_idx, :, :);

% 2. Интерполяция (Upsampling) до 1000 Гц
% interp1 отлично работает с многомерными массивами вдоль 1-го измерения
locations_upsampled = interp1(x_unique, v_unique, xq, 'pchip', 'extrap'); 
% Результат: матрица [507008 x 2 x 21]

% 3. Вычисление скоростей (на частоте 1000 Гц)
% Находим разницу координат за 1 миллисекунду
delta_pos = diff(locations_upsampled, 1, 1); % [507007 x 2 x 21]

% Извлекаем смещения по осям X и Y
dx = squeeze(delta_pos(:, 1, :)); % [507007 x 21]
dy = squeeze(delta_pos(:, 2, :)); % [507007 x 21]

% Считаем модуль скорости (теорема Пифагора)
speeds_upsampled = sqrt(dx.^2 + dy.^2); % [507007 x 21]

% Так как delta_pos рассчитана за 1 мс, текущая скорость имеет размерность "пиксели/мс".
% Переведем ее в более понятные "пиксели/секунду", умножив на частоту дискретизации
speeds_upsampled_sec = speeds_upsampled * init_data.samplerate;

%%
% 1. Инициализация
state_arr = init_data.state; 
n_total = length(state_arr);
n_points = 10000;

% Создаем независимые логические маски (каждая размером с запись)
masks = cell(5, 1);
for c = 1:5
    masks{c} = false(n_total, 1); 
end

% 2. Твои индексы переключений
change_idx = find(diff(state_arr) ~= 0) + 1;
change_idx = unique([change_idx; 55121]); 
change_idx = sort(change_idx);

for i = 1:length(change_idx)
    start_pt = change_idx(i);
    if start_pt == 55121
        start_pt = start_pt + 1;
    end
    end_pt = start_pt + n_points - 1;
        
    current_class = state_arr(start_pt);
    
    if current_class >= 1 && current_class <= 5
        masks{current_class}(start_pt:end_pt) = true;
    end
end

%%
figure
plot(speeds_upsampled_sec)
hold on
plot(state_arr*200)
plot(masks{5}*5000)

%%
X = data.epochs;
X = X(:,1:55,:);

Fs = data.sfreq;
Fmin = 8;
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
    
    Epochs(:,:,ep_idx) = Epfilt(Fs+1:end-Fs-1,:);
end

%%
Ws = 1;
Ss = 1;

X_epochs = [];
for ep_idx=1:size(Epochs,3)
    ep_idx
    xeps = epoch_data(Epochs(:,:,ep_idx),Fs,Ws,Ss);
    X_epochs = cat(3, X_epochs, xeps);
end

%%
n_points = 10000; 
n_fingers = 21;   
cond = 1;         

mask_eeg = data.events(:,3) == cond;
Epochs_cond = Epochs(:,:,mask_eeg); 

mask_vars = masks{cond};
speeds_vec = speeds_upsampled_sec(mask_vars, :); 

n_trials = 10;
speeds_epochs = []; m = 1:n_points;
for i=1:10
    speeds_epochs(:,:,i) = speeds_vec(m,:);
    m = m + n_points;
end

% figure
% plot(speeds_vec)
% 
% figure
% plot(speeds_epochs(:,:,2))

Ws = 0.5;
Ss = Ws / 2;

X_epo = []; Z_epo = [];
for ep_idx=1:size(Epochs_cond,3)
    ep_idx
    xeps = epoch_data(Epochs_cond(:,:,ep_idx),Fs,Ws,Ss);
    X_epo = cat(3, X_epo, xeps);

    zeps = epoch_data(speeds_epochs(:,:,ep_idx),Fs,Ws,Ss);
    Z_epo = cat(3, Z_epo, zeps);
end
size(X_epo)
size(Z_epo)
Z_epo = squeeze(mean(Z_epo,1));
size(Z_epo)

%%
Covs = [];
for i=1:size(X_epochs,3)
    i
    Covs(:,:,i) = cov(X_epochs(:,:,i));
end
Tcovs = Tangent_space(Covs);           

%%
DistsT = squareform(pdist(Tcovs', 'euclidean'));

%%
N_neigb = 5;
gamma = 10;

W = exp(-(DistsT.^2) / (2 * gamma^2));
W = W - diag(diag(W));

W_n = zeros(size(DistsT));
for i=1:size(DistsT,1)
    [mvals, mids] = sort(W(i,:),'descend');
    W_n(i,mids(2:1+N_neigb)) = mvals(2:1+N_neigb);
end
W_n = (W_n + W_n') / 2;

D = diag(sum(W_n,2));
L = D - W_n;

[U,S] = eigs(L, D, 1+10,'smallestreal');
S = diag(S);

U = U(:,2:end);

scatter3(U(:,1),U(:,2),U(:,3))

%%
Z_epo = (Z_epo - mean(Z_epo,2)) ./ std(Z_epo,[],2);
[W, A, Vf, Vz, corrs, Feat, Epochs_cov, eigenvalues] = espoc(X_epo, Z_epo);

figure
stem(corrs');

%%
gl_c = 1;
com_idx = 55;
w = W(1, :, com_idx)';
a = A(1, :, com_idx)';

Z_epo_pr = Vz(:,gl_c)' * Z_epo;

env = [];
for i=1:size(Epochs_cov,3)
    env(i) = w' * Epochs_cov(:,:,i) * w;
end

Z_epo_pr = (Z_epo_pr - mean(Z_epo_pr,2)) ./ std(Z_epo_pr,[],2);
env = (env - mean(env)) / std(env);
figure
plot(env)
hold on
plot(Z_epo_pr)

corr(env',Z_epo_pr')

%%
% Делаем максимальное абсолютное значение паттерна положительным
[~, imax] = max(abs(a));
sign_flip = sign(a(imax));
a = a * sign_flip;

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

xlabel('X (Right)');
ylabel('Y (Nose)');
zlabel('Z (Up)');

view(-45, 30);
xlim([cx-r, cx+r]); ylim([cy-r, cy+r]); zlim([cz-r, cz+r]);
hold off

