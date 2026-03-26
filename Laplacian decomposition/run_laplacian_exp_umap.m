close all
clear
clc

ft_path = 'C:\Users\ansbel\Documents\2Git\fieldtrip';

if ~exist('ft_defaults','file')
    addpath(ft_path);
end

ft_defaults;

%%
sub_path = 'Tumyalis_music_epochs.fif';

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
conditions = {'(1) RS EC 1', '(2) RS EO 1', '(3) 2Hz', '(4) 05Hz', '(5) 4Hz', '(6) 1Hz', '(7) 3Hz', ...
              '(8) NoRy 1','(9) Waltz 1','(10) Waltz 2','(11) NoRy 2','(12) NoRy 3','(13) Waltz 3', ...
              '(14) NoRy 4','(15) Waltz 4','(16) NoRy 5','(17) Waltz 5','(18) RS EC 2','(19) RS EO 2', ...
              '(20) Waltz 6','(21) Waltz 7','(22) Waltz 8'};

%% =====================================================================
[b,a] = butter(3,[15,25]/(Fs/2));   
n_channels = 38;                    

Xfilt = [];
Epfilt = [];
Epochs_filt = [];

for i = 1:numel(Xinf.trial)
    i
    Ep_raw = Xinf.trial{i}(1:n_channels,:);            
    Epfilt  = filtfilt(b,a,Ep_raw')';    
    Epfilt = Epfilt(:,Fs/2:end-Fs/2);    

    Xfilt = cat(2,Xfilt,Epfilt);         
    Epochs_filt(:,:,i) = Epfilt;         
end

%% =====================================================================
[Upca,S,~] = svd(Xfilt,'econ');           
S = diag(S);

tol = max(size(Xfilt)) * eps(S(1));
r = sum(S > tol);

ve = S.^2;
var_explained = cumsum(ve) / sum(ve);
var_explained(end) = 1;

n_components = find(var_explained>=1, 1);
n_components = max(min(n_components, r), 1);

Upca = Upca(:,1:n_components);               

Epfilt_pca = [];
for i = 1:size(Epochs_filt,3)
    Epfilt_pca(:,:,i) = Upca'*Epochs_filt(:,:,i);
end

Xfiltpca = Upca'*Xfilt;

%% =====================================================================
Wsize = 2;  
Ssize = 0.5;  

X_epo = []; time = [];
for i=1:size(Epfilt_pca,3)
    i

    ep_wins = epoch_data(Epochs_filt(:,:,i)', Fs, Wsize, Ssize);
    X_epo = cat(3,X_epo,ep_wins); 
    
end

Covs = []; Covs_vec = [];
for i=1:size(X_epo,3)
    C = cov(X_epo(:,:,i));
    Covs(:,:,i) = C;
    Covs_vec(i,:) = cov2upper(C);
end

Cm = cov(Xfilt');

%% =====================================================================
Wsize = 2;  
Ssize = 0.5;

X_epo = []; time = [];
for i=1:size(Epfilt_pca,3)
    i
    
    ep_wins = epoch_data(Epfilt_pca(:,:,i)', Fs, Wsize, Ssize);
    X_epo = cat(3,X_epo,ep_wins); 
    
end

Covs_pca = []; Covs_vec_pca = [];
for i=1:size(X_epo,3)
    C = cov(X_epo(:,:,i));
    Covs_pca(:,:,i) = C;
    Covs_vec_pca(i,:) = cov2upper(C);
end

Cm_pca = riemann_mean(Covs_pca);
Tcovs = Tangent_space(Covs_pca,Cm_pca);          
N_epoch_trial = size(ep_wins,3);

%% 1. Построение матрицы эталонных вероятностей P (из Tcovs)
disp('Вычисляем матрицу притяжения P...');
N_epochs = size(Tcovs, 2);
N_channels = size(Covs, 1);

% Вычисляем попарные евклидовы расстояния в касательном пространстве
% (Евклидово расстояние в Tcovs эквивалентно Риманову геодезическому в Covs)
D_high = squareform(pdist(Tcovs')); 

% Эвристика для ширины ядра (можно настроить, например, как 10-15% от max(D))
sigma = median(D_high(:)) / 2; 

% Переводим расстояния в вероятности (Gaussian Kernel)
P = exp(-(D_high.^2) / (2 * sigma^2));
P(1:N_epochs+1:end) = 0; % Обнуляем диагональ (вероятность точки быть соседом самой себе = 0)

% Симметризуем и нормируем, чтобы сумма всех P_ij была равна 1
P = P + P';
P = P / sum(P(:));

%% 2. Инициализация параметров
w = randn(N_channels, 1);
w = w / norm(w); % Стартуем с фильтра единичной длины

epochs_gd = 800;      % Количество итераций
learning_rate = 1000;  % В t-SNE/UMAP learning rate обычно большой (от 10 до 1000)
momentum = 0.8;       % Накопление импульса
velocity_w = zeros(N_channels, 1);

loss_history = zeros(epochs_gd, 1);

%% 3. Оптимизация w (UMAP Cross-Entropy)
disp('Запуск градиентного спуска (Истинный UMAP-loss)...');

for iter = 1:epochs_gd
    
    % --- 1. Прямой проход (Векторизованно для скорости) ---
    % Cw_all размер: [N_channels x N_epochs]
    Cw_all = squeeze(pagemtimes(Covs, w)); 
    % Мощность z_i = w^T C_i w
    z = sum(w .* Cw_all, 1)'; 
    z = z / std(z); % Стабилизация масштаба
    
    % --- 2. Вероятности Q (БЕЗ глобальной нормализации) ---
    Z_diff = z - z';       % (z_i - z_j)
    D_sq = Z_diff.^2;      % Квадраты расстояний
    
    % Семейство кривых UMAP (при a=1, b=1)
    Q = 1 ./ (1 + D_sq);   
    Q(1:N_epochs+1:end) = 0; % Зануляем диагональ
    
    % Вычисление кросс-энтропии (UMAP Loss, Eq. 3 в статье)
    P_safe = max(P, 1e-12);
    Q_safe = max(Q, 1e-12);
    term1 = P_safe .* log(P_safe ./ Q_safe);                     % Притяжение
    term2 = (1 - P_safe) .* log((1 - P_safe) ./ (1 - Q_safe));   % Отталкивание
    loss_history(iter) = sum(term1(:) + term2(:));
    
    % --- 3. Истинные градиенты UMAP по переменной z ---
    % Сила притяжения (тянет соседей друг к другу)
    % grad_attr = -2 * P_{ij} * (z_i - z_j) * Q_{ij}
    attr_forces = P .* Q; 
    
    % Сила отталкивания (расталкивает не-соседей)
    % grad_rep = 2 * (1 - P_{ij}) * (z_i - z_j) * Q_{ij} / (D_sq + eps)
    rep_forces = (1 - P) .* (Q ./ (D_sq + 1e-8)); 
    
    % Итоговый градиент (скалярные силы для каждой эпохи)
    % Знак минус у attr_forces и плюс у rep_forces учитываются при вычитании в Z_diff
    Stiffness = attr_forces - rep_forces; 
    grad_z = 2 * sum(Stiffness .* Z_diff, 2); % Вектор [N_epochs x 1]
    
    % --- 4. Цепное правило (Проецируем UMAP-силы на веса электродов) ---
    % grad_w = 2 * sum_i ( C_i * w * grad_z_i )
    grad_w = 2 * Cw_all * grad_z; 
    
    % --- 5. Шаг оптимизатора ---
    velocity_w = momentum * velocity_w - learning_rate * grad_w;
    w = w + velocity_w;
    w = w / norm(w); % Проекция
    
    if mod(iter, 10) == 0
        fprintf('Итерация %d | UMAP CE Loss: %.4f\n', iter, loss_history(iter));
    end
end

%% 4. Визуализация
figure;
subplot(2,1,1);
plot(loss_history, 'LineWidth', 2);
title('Сходимость: KL-дивергенция');
xlabel('Итерация'); ylabel('Loss');
grid on;

% Вычисляем итоговую огибающую мощности
Env_final = zeros(1, N_epochs);
for i = 1:N_epochs
    Env_final(i) = w' * Covs(:,:,i) * w;
end
Env_final = (Env_final - mean(Env_final)) / std(Env_final);

subplot(2,1,2);
plot(Env_final, 'LineWidth', 1.5, 'Color', '#D95319');
title('Огибающая мощности найденного источника (UMAP -> w)');
xlabel('Индекс эпохи'); ylabel('Мощность (z-score)');
grid on;

%%
a = Cm * w;
topo.avg   = a;
ft_topoplotER(cfg, topo);

%%