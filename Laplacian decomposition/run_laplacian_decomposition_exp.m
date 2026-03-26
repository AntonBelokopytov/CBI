close all
clear
clc

ft_path = 'D:\OS(CURRENT)\scripts\2Git\fieldtrip\';

if ~exist('ft_defaults','file')
    addpath(ft_path);
end

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
X = Xinf.trial{1};

%%
[b,a] = butter(3,[8,12]/(Fs/2));   
Xfilt = filtfilt(b,a,X')';    

%% Построение разреженного графа соседей без полной матрицы расстояний
X_trans = Xfilt'; 
n = size(X, 2);
k = 500; 

disp('Ищем ближайших соседей...');
% Функция knnsearch (Statistics and Machine Learning Toolbox) 
% находит k+1 соседей напрямую. Мы ищем k+1, так как первой 
% найденной точкой всегда будет сама точка (расстояние = 0).
[idx, D_knn] = knnsearch(X_trans, X_trans, 'K', k + 1, 'Distance', 'euclidean');

% Удаляем саму точку из списка соседей
idx(:, 1) = []; 
D_knn(:, 1) = []; 

disp('Вычисляем веса и строим разреженную матрицу...');
% Эвристика для ширины окна ядра (sigma)
sigma = median(D_knn(:)); 

% Вычисляем веса по ядру теплопроводности
weights = exp(-(D_knn.^2) / (2 * sigma^2));

% Подготавливаем индексы для создания разреженной матрицы
row_idx = repmat((1:n)', 1, k);

% Создаем разреженную матрицу (sparse matrix). 
% Она займет всего несколько мегабайт вместо 26 ГБ.
W = sparse(row_idx(:), idx(:), weights(:), n, n);

% Делаем матрицу симметричной (если точка A - сосед B, то W(A,B) = W(B,A))
W = max(W, W');

disp('Граф смежности успешно построен!');

%% Вычисление Лапласиана
D_deg = spdiags(sum(W, 2), 0, n, n); % Разреженная диагональная матрица
L = D_deg - W; % Разреженный Лапласиан

%%
A = X * L * X'; 
B = X * D_deg * X';
A = (A + A') / 2;
B = (B + B') / 2;

lambda_reg = 1e-6; 
B = B + eye(size(B)) * lambda_reg * trace(B);

%%
[W_all, Eigenvalues_matrix] = eig(A, B);
eigenvalues = diag(Eigenvalues_matrix);
% Наша цель - argmin(w), поэтому сортируем значения по ВОЗРАСТАНИЮ.
% Фильтры, соответствующие наименьшим собственным значениям, лучше всего 
% сохраняют локальную структуру графа.
[eigenvalues, sort_idx] = sort(eigenvalues, 'ascend');
stem(eigenvalues)

Filters = W_all(:, sort_idx);

Sources = Filters' * X;

%%
Patterns = X * Sources' / (Sources * Sources');

%%
figure
plot(Sources(38,:))

%%
topo.avg   = Patterns(:,38);
ft_topoplotER(cfg, topo);

%%
