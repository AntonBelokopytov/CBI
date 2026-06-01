%% =====================================================================
% BENCHMARK: THREADS VS PROCESSES (TIME PER ITERATION)
% =====================================================================
chs = size(Xfiltpca, 1);
samples_per_epoch = size(Xfiltpca, 2) / nEpochs;

% Задаем нагрузку: каждый воркер всегда делает ровно 10 итераций
iters_per_worker = 20; 
worker_list = 1:8; 
pool_types = {'Threads', 'Processes'};

% Матрица для сохранения эффективного времени 1 итерации
time_per_iter = zeros(length(pool_types), length(worker_list));

disp('Starting Benchmark...');

for p = 1:length(pool_types)
    pType = pool_types{p};
    
    for w = 1:length(worker_list)
        nw = worker_list(w);
        
        % Общее NMC всегда кратно числу воркеров
        NMC = iters_per_worker * nw; 
        
        delete(gcp('nocreate'));
        parpool(pType, nw);
        
        fprintf('Testing %s with %d workers (Total NMC = %d)...\n', pType, nw, NMC);
        
        batch_corrmax = cell(1, nw);
        batch_corrmin = cell(1, nw);
        
        tic;
        parfor (worker_id = 1:nw, nw)
            
            % Массивы фиксированного размера без сложной математики линспейсов
            loc_corrmax = zeros(3, iters_per_worker);
            loc_corrmin = zeros(3, iters_per_worker);
            
            for b = 1:iters_per_worker
                r_idx = fix(rand * size(Xfiltpca, 2));
                
                XCirc = circshift(Xfiltpca, [0, r_idx]);
                mask_ts_shifted = circshift(mask_ts, [0, r_idx]);
                
                Eps_circ = reshape(XCirc, chs, samples_per_epoch, nEpochs);
                mask_ts_shifted_eps = reshape(mask_ts_shifted, samples_per_epoch, nEpochs);
                XCirc = []; mask_ts_shifted = [];
                
                X_test_cell = cell(1, nEpochs);
                for j = 1:nEpochs
                    ep_wins = epoch_data(Eps_circ(:,:,j)', Fs, Wsize, Ssize);
                    mask_ep_wins = epoch_data(double(mask_ts_shifted_eps(:,j)), Fs, Wsize, Ssize);
                    valid_windows = all(mask_ep_wins, 1);
                    X_test_cell{j} = ep_wins(:,:,valid_windows);
                end
                Eps_circ = []; mask_ts_shifted_eps = [];
                
                X_test = cat(3, X_test_cell{:});
                
                neps = min(size(X_test,3), size(R,1));
                [~,~,~,~,corrs_perm] = espoc(X_test(:,:,1:neps), R(1:neps,:)');
                
                loc_corrmax(:, b) = max(corrs_perm, [], 2);
                loc_corrmin(:, b) = min(corrs_perm, [], 2); % Исправлен max на min
            end
            
            batch_corrmax{worker_id} = loc_corrmax;
            batch_corrmin{worker_id} = loc_corrmin;
        end
        total_time = toc;
        
        % Эффективное время ОДНОЙ итерации.
        % Делим общее время работы пула на число итераций, которое сделал один воркер 
        % (поскольку они работали параллельно).
        time_per_iter(p, w) = total_time / iters_per_worker;
        
        fprintf('Total time: %.2f sec | Effective time per iter: %.2f sec\n\n', total_time, time_per_iter(p, w));
        
        corrmax = cat(2, batch_corrmax{:});
        corrmin = cat(2, batch_corrmin{:});
    end
end
delete(gcp('nocreate'));

%% =====================================================================
% PLOT THE RESULTS
% =====================================================================
figure; set(gcf, 'Color', 'w');
plot(worker_list, time_per_iter(1, :), '-o', 'LineWidth', 2, 'MarkerSize', 8, 'DisplayName', 'Threads');
hold on;
plot(worker_list, time_per_iter(2, :), '-s', 'LineWidth', 2, 'MarkerSize', 8, 'DisplayName', 'Processes');
grid on;
xlabel('Number of Workers');
ylabel('Effective Time for 1 Iteration (seconds)');
title('Execution Time of a Single MC Iteration vs Number of Workers');
legend('Location', 'best');
xticks(worker_list);


%% =====================================================================
% PLOT 2: THROUGHPUT (ITERATIONS PER SECOND)
% =====================================================================
% Вычисляем пропускную способность: (кол-во воркеров) / (эффективное время 1 итерации)
throughput = zeros(size(time_per_iter));
for p = 1:size(time_per_iter, 1)
    for w = 1:length(worker_list)
        throughput(p, w) = worker_list(w) / time_per_iter(p, w);
    end
end

figure; set(gcf, 'Color', 'w');
plot(worker_list, throughput(1, :), '-o', 'LineWidth', 2, 'MarkerSize', 8, 'DisplayName', 'Threads');
hold on;
plot(worker_list, throughput(2, :), '-s', 'LineWidth', 2, 'MarkerSize', 8, 'DisplayName', 'Processes');

% Настройка визуализации
grid on;
xlabel('Number of Workers');
ylabel('Throughput (iterations per second)');
title('System Throughput vs Number of Workers (Higher is Better)');
legend('Location', 'best');
xticks(worker_list);

% Добавляем маркер максимума для Тредов, чтобы наглядно видеть "Sweet Spot"
[max_thr, max_idx] = max(throughput(1, :));
plot(worker_list(max_idx), max_thr, 'rp', 'MarkerSize', 14, 'MarkerFaceColor', 'r', 'HandleVisibility', 'off');
text(worker_list(max_idx), max_thr * 1.05, sprintf('  Peak: %.2f iter/s', max_thr), 'Color', 'r', 'FontWeight', 'bold');