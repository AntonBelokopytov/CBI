%% =====================================================================
% PERMUTATION TEST (CIRCULAR TIME SHIFTS)
% =====================================================================
chs = size(Xfiltpca, 1);
samples_per_epoch = size(Xfiltpca, 2) / nEpochs; 
corrmax = zeros(3, nMC);
corrmin = zeros(3, nMC);

disp('Running Permutation Test...');
parfor i = 1:nMC
    i
    r_idx = fix(rand * size(Xfiltpca, 2));
    XCirc = circshift(Xfiltpca, [0, r_idx]);
    mask_ts_shifted = circshift(mask_ts, [0, r_idx]);
    
    Eps_circ = reshape(XCirc, chs, samples_per_epoch, nEpochs);
    mask_ts_shifted_eps = reshape(mask_ts_shifted, samples_per_epoch, nEpochs);
    
    X_test = [];
    for j = 1:nEpochs
        ep_wins = epoch_data(Eps_circ(:,:,j)', Fs, Wsize, Ssize);
        mask_ep_wins = epoch_data(double(mask_ts_shifted_eps(:,j)), Fs, Wsize, Ssize);
        valid_windows = all(mask_ep_wins, 1);
        X_test = cat(3, X_test, ep_wins(:,:,valid_windows));
    end
    
    neps = min(size(X_test,3),size(R,1));
    [~,~,~,~,corrs_perm] = espoc(X_test(:,:,1:neps), R(1:neps,:)'); 
    corrmax(:,i) = max(corrs_perm,[],2);
    corrmin(:,i) = min(corrs_perm,[],2);
end

% --- ФИЛЬТРАЦИЯ НУЛЕВЫХ ИТЕРАЦИЙ ---
% Находим столбцы, где значения не равны строго 0 
% (eSPoC корреляция практически никогда не бывает ровно 0.0000)
valid_iters = max(corrmax, [], 1) ~= 0;
corrmax_valid = corrmax(:, valid_iters);
corrmin_valid = corrmin(:, valid_iters);

actual_nMC = size(corrmax_valid, 2);
disp(['Фактически выполнено итераций: ', num2str(actual_nMC)]);

if actual_nMC < 20
    warning('Слишком мало итераций для надежной оценки альфа = 0.05');
end

if actual_nMC > 0
    % Сортируем только валидные значения
    corrmax1 = sort(max(corrmax_valid, [], 1), 'descend');
    corrmin1 = sort(min(corrmin_valid, [], 1));
    
    alpha = 0.05;
    
    % Расчет максимального порога с защитой от выхода за пределы
    i = 1; 
    while (1 - sum(corrmax1(i) > corrmax1) / actual_nMC) <= alpha && i < actual_nMC
        i = i + 1; 
    end
    max_val = corrmax1(i);
    
    % Расчет минимального порога с защитой от выхода за пределы
    i = 1; 
    while (1 - sum(corrmin1(i) < corrmin1) / actual_nMC) <= alpha && i < actual_nMC
        i = i + 1; 
    end
    min_val = corrmin1(i);
else
    disp('Нет данных для расчета порогов.');
    max_val = NaN;
    min_val = NaN;
end