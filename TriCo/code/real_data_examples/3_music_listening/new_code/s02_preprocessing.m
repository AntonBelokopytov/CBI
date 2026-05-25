% =====================================================================
% BANDPASS FILTERING & CONTINUOUS MASKING
% =====================================================================
[b,a] = butter(3, freq_band/(Fs/2));   
Xfilt = [];
Epochs_filt = [];
time_series = []; 
en_t = 0;

for i = 1:numel(Xinf.trial)
    Ep_raw = Xinf.trial{i}(1:n_channels,:);            
    Epfilt = filtfilt(b,a,Ep_raw')';         
    Epfilt = Epfilt(:,Fs/2:end-Fs/2);        
    ep_time = 1:size(Ep_raw,2); 
    
    time_series(:,i) = en_t + ep_time; 
    en_t = en_t + ep_time(end);
    
    Xfilt = cat(2,Xfilt,Epfilt);             
    Epochs_filt(:,:,i) = Epfilt;             
end
time_series = time_series(Fs/2:end-Fs/2,:);
time_series_raw = reshape(time_series,1,[]);

% Маска плохих сэмплов (1 - хороший, 0 - артефакт)
mask_ts = true(1,size(time_series_raw,2));
for i=1:size(BADS,1)
    bad_st = BADS(i,1);
    bad_en = bad_st + BADS(i,2);
    bad_idx = (time_series_raw >= bad_st) & (time_series_raw <= bad_en);
    mask_ts(bad_idx) = false;
end

% Сразу возвращаем маску в 2D формат (Сэмплы х Условия) для нарезки
mask_ts_mat = reshape(mask_ts, size(time_series,1), size(time_series,2));

% =====================================================================
% SVD AND PCA DIMENSIONALITY REDUCTION
% =====================================================================
[U,S,~] = svd(Xfilt(:,mask_ts),'econ');
S = diag(S);
tol = max(size(Xfilt)) * eps(S(1));
r = sum(S > tol);
ve = S.^2;
var_explained = cumsum(ve) / sum(ve);
var_explained(end) = 1;
n_components = find(var_explained>=1, 1);
n_components = max(min(n_components, r), 1);
U = U(:,1:n_components);               

Epfilt_pca = zeros(n_components, size(Epochs_filt,2), size(Epochs_filt,3));
for i = 1:size(Epochs_filt,3)
    Epfilt_pca(:,:,i) = U'*Epochs_filt(:,:,i);
end
Xfiltpca = U'*Xfilt;

% =====================================================================
% EPOCH SEGMENTATION & ARTIFACT MASKING (УПРОЩЕННАЯ ЛОГИКА)
% =====================================================================
X_epo = []; 
cond_idx_epochs = []; 
ep_mask = []; % Собираем маску прямо в цикле нарезки

for i=1:size(Epfilt_pca,3)
    % 1. Нарезаем ЭЭГ-данные
    ep_wins = epoch_data(Epfilt_pca(:,:,i)', Fs, Wsize, Ssize);
    X_epo = cat(3, X_epo, ep_wins); 
    
    % 2. Нарезаем маску точно так же
    m_wins = epoch_data(double(mask_ts_mat(:,i)), Fs, Wsize, Ssize);
    
    % 3. Окно валидно, если в нём нет ни одного артефактного сэмпла (нуля)
    valid_windows = squeeze(all(m_wins, 1))';
    ep_mask = [ep_mask, valid_windows];
    
    cond_idx_epochs = [cond_idx_epochs, repmat(i, 1, size(ep_wins, 3))];
end

ep_mask = logical(ep_mask);

% Ковариационные матрицы
Covs = zeros(size(X_epo,2), size(X_epo,2), size(X_epo,3)); 
for i=1:size(X_epo,3)
    Covs(:,:,i) = cov(X_epo(:,:,i));
end
Tcovs = Tangent_space(Covs);           
N_epoch_trial = size(ep_wins,3);

% Таймлайны для отрисовки
valid_cond_idx = cond_idx_epochs(ep_mask);
boundaries = find(diff(valid_cond_idx) > 0);
ticks = [0, boundaries, length(valid_cond_idx)];
valid_Covs = Covs(:,:,ep_mask);