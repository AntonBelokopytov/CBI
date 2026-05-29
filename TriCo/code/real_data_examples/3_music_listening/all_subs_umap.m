close all
clear
clc

% =====================================================================
% SETUP PATHS & DIRS
% =====================================================================
ft_path = 'C:\Users\anton\Documents\GitHub\CBI\site-packages\fieldtrip';
if ~exist('ft_defaults','file')
    addpath(ft_path);
end
ft_defaults;

% Базовая директория с данными
data_dir = 'D:/OS(CURRENT)/scripts/2Git/TriCo/data/external/music_listening';

% Папка для сохранения результатов
results_dir = fullfile(data_dir, 'Results');
if ~exist(results_dir, 'dir')
    mkdir(results_dir);
end

% Загружаем структуру BADS один раз для всех
BADS_all = load(fullfile(data_dir, 'BADS.mat')).BADS;

% Ищем все fif файлы
fif_files = dir(fullfile(data_dir, '*_music_epochs.fif'));

conditions = {'(1) RS EC 1', '(2) RS EO 1', '(3) 2Hz', '(4) 05Hz', '(5) 4Hz', '(6) 1Hz', '(7) 3Hz', ...
              '(8) NoRy 1','(9) Waltz 1','(10) Waltz 2','(11) NoRy 2','(12) NoRy 3','(13) Waltz 3', ...
              '(14) NoRy 4','(15) Waltz 4','(16) NoRy 5','(17) Waltz 5','(18) RS EC 2','(19) RS EO 2', ...
              '(20) Waltz 6','(21) Waltz 7','(22) Waltz 8'};

%% =====================================================================
% MAIN LOOP OVER SUBJECTS
% =====================================================================
for sub_idx = 1:length(fif_files)
    
    file_name = fif_files(sub_idx).name;
    sub_path = fullfile(data_dir, file_name);
    
    % Достаем имя испытуемого (все, что до первого '_')
    name_parts = strsplit(file_name, '_');
    sub_name = name_parts{1};
    
    fprintf('=======================================================\n');
    fprintf('PROCESSING SUBJECT: %s (%d/%d)\n', sub_name, sub_idx, length(fif_files));
    fprintf('=======================================================\n');
    
    % Создаем папку для графиков текущего испытуемого
    sub_save_dir = fullfile(results_dir, sub_name);
    if ~exist(sub_save_dir, 'dir')
        mkdir(sub_save_dir);
    end
    
    % Достаем BADS конкретного испытуемого
    if isfield(BADS_all, sub_name)
        curr_bads_sec = BADS_all.(sub_name);
    else
        warning('BADS для %s не найдены! Используем пустой массив.', sub_name);
        curr_bads_sec = [];
    end
    
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
    
    BADS = fix(curr_bads_sec * Fs);
    
    % =====================================================================
    % BANDPASS FILTERING
    % =====================================================================
    [b,a] = butter(3,[15,25]/(Fs/2));   
    n_channels = 38;                    
    Xfilt = [];
    Epochs_filt = [];
    time_series = []; en_t = 0;
    
    for i = 1:numel(Xinf.trial)
        Ep_raw = Xinf.trial{i}(1:n_channels,:);            
        Epfilt  = filtfilt(b,a,Ep_raw')';    
        Epfilt = Epfilt(:,Fs/2:end-Fs/2);    
        ep_time = 1:size(Ep_raw,2); 
        time_series(:,i) = en_t + ep_time; 
        en_t = en_t + ep_time(end);
        Xfilt = cat(2,Xfilt,Epfilt);         
        Epochs_filt(:,:,i) = Epfilt;         
    end
    time_series = time_series(Fs/2:end-Fs/2,:);
    time_series_raw = reshape(time_series,1,[]);
    
    mask_ts = true(1,size(time_series_raw,2));
    for i=1:size(BADS,1)
        bad_st = BADS(i,1);
        bad_en = bad_st + BADS(i,2);
        bad_idx = (time_series_raw >= bad_st) & (time_series_raw <= bad_en);
        mask_ts(bad_idx) = false;
    end
    
    % =====================================================================
    % SVD AND PCA
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
    
    Epfilt_pca = [];
    for i = 1:size(Epochs_filt,3)
        Epfilt_pca(:,:,i) = U'*Epochs_filt(:,:,i);
    end
    Xfiltpca = U'*Xfilt;
    
    % =====================================================================
    % EPOCH SEGMENTATION
    % =====================================================================
    Wsize = 2;  
    Ssize = 0.5;  
    X_epo = []; time = [];
    time_series_epochs = [];
    
    for i=1:size(Epfilt_pca,3)
        ep_wins = epoch_data(Epfilt_pca(:,:,i)', Fs, Wsize, Ssize);
        X_epo = cat(3,X_epo,ep_wins); 
        
        ts_wins = epoch_data(time_series(:,i), Fs, Wsize, Ssize);
        time_series_epochs = cat(2,time_series_epochs,ts_wins); 
        timeline = 0.5 + ( Wsize/2:Ssize:(size(ep_wins,3)*Ssize+Ssize) );
        if i>1
            timeline = timeline + time(end) + Wsize-Ssize;
        end
        time = [time,timeline];
    end
    
    Covs = []; 
    for i=1:size(X_epo,3)
        Covs(:,:,i) = cov(X_epo(:,:,i));
    end
    
    % Проекция в тангенциальное пространство с новой стабильной функцией
    Tcovs = Tangent_space(Covs);           
    N_epoch_trial = size(ep_wins,3);
    
    ep_mask = true(1,size(X_epo,3));
    for ep_idx=1:size(time_series_epochs,2)
        ep_st = time_series_epochs(1,ep_idx);
        ep_en = time_series_epochs(end,ep_idx);
        for i=1:size(BADS,1)
            bad_st = BADS(i,1);
            bad_en = BADS(i,1) + BADS(i,2);
            if (bad_st <= ep_en) && (bad_en >= ep_st)
                ep_mask(ep_idx) = false;
            end
        end
    end
    
    % =====================================================================
    % UMAP EMBEDDING
    % =====================================================================
    clear u
    u = UMAP("n_neighbors",20,"n_components",3,"min_dist",0);
    u.metric = 'euclidean';
    u.target_metric = 'euclidean';
    R = u.fit_transform(Tcovs');
    Rmean = R - mean(R,1);
    
    % =====================================================================
    % VISUALIZE UMAP (FIGURE 1)
    % =====================================================================
    fig1 = figure('Visible', 'off');
    set(fig1, 'Color', 'w');
    scatter3(R(:,1),R(:,2),R(:,3));
    title(sprintf('UMAP: %s', sub_name), 'Interpreter', 'none');
    xlabel('UMAP component 1'); ylabel('UMAP component 2'); zlabel('UMAP component 3');
    exportgraphics(fig1, fullfile(sub_save_dir, '01_UMAP_3D.png'), 'Resolution', 300);
    
    % =====================================================================
    % Temporal evolution of UMAP components (FIGURE 2)
    % =====================================================================
    fig2 = figure('Visible', 'off');
    plot(R)                          
    set(fig2, 'Color', 'w');
    tstep = 235;                     
    ticks = 0:tstep:size(R,1);
    times = 0:120:22*120;            
    xticks(ticks); xticklabels(times); xlim([0,size(R,1)]);
    ylabel('UMAP component coordinate'); xlabel('time, sec');
    legend({'component 1', 'component 2', 'component 3'});
    title(sprintf('UMAP Dynamics: %s', sub_name), 'Interpreter', 'none');
    exportgraphics(fig2, fullfile(sub_save_dir, '02_UMAP_Dynamics.png'), 'Resolution', 300);
    
    % =====================================================================
    % eSPoC COMPUTATION (FIGURE 3)
    % =====================================================================
    [W, A, Vf, Vz, corrs, VecCov, Epochs_cov] = espoc(X_epo, R');
    
    fig3 = figure('Visible', 'off');
    set(fig3,'Color','w');
    stem(corrs','LineWidth',1.5);
    grid on
    xlabel('Local component index'); ylabel('Correlation');
    title(sprintf('eSPoC correlation values: %s', sub_name), 'Interpreter', 'none');
    legend({'Global 1','Global 2','Global 3'}, 'Location','best');
    xlim([1 size(corrs,2)]);
    exportgraphics(fig3, fullfile(sub_save_dir, '03_eSPoC_Correlations.png'), 'Resolution', 300);
    
    % =====================================================================
    % Canonical projections (FIGURE 4)
    % =====================================================================
    gl_src = Vf' * VecCov;
    emb_can_pr = Vz' * R';
    
    fig4 = figure('Visible', 'off', 'Position', [100 100 800 800]); 
    set(fig4,'Color','w');
    t = tiledlayout(3,2);
    title(t, sprintf('Canonical Projections: %s', sub_name), 'Interpreter', 'none');
    
    nexttile(1,[3,1])
    x = Vz(:,1)' * Rmean'; y = Vz(:,2)' * Rmean'; z = Vz(:,3)' * Rmean';
    num_clusters = 22;
    cmap = jet(num_clusters);
    ccx=[]; ccy=[]; ccz=[];
    mask = 1:N_epoch_trial;
    for i = 1:num_clusters
        if mask(end) <= numel(x)
            sc_x = x(mask); sc_y = y(mask); sc_z = z(mask);
        else
            sc_x = x(mask(1):end); sc_y = y(mask(1):end); sc_z = z(mask(1):end);
        end
        mask = mask + N_epoch_trial;
        ccx = [ccx, mean(sc_x)]; ccy = [ccy, mean(sc_y)]; ccz = [ccz, mean(sc_z)];
    end
    plot3(ccx, ccy, ccz, 'k', 'LineWidth', 1);
    hold on; grid on
    legend_handles = gobjects(num_clusters,1);
    mask = 1:N_epoch_trial;
    for i = 1:num_clusters
        if mask(end) <= numel(x)
            sc_x = x(mask); sc_y = y(mask); sc_z = z(mask);
        else
            sc_x = x(mask(1):end); sc_y = y(mask(1):end); sc_z = z(mask(1):end);
        end
        mask = mask + N_epoch_trial;
        cx = mean(sc_x); cy = mean(sc_y); cz = mean(sc_z);
        scatter3(sc_x, sc_y, sc_z, 10, repmat(cmap(i,:), length(sc_x), 1), 'filled', 'MarkerFaceAlpha', 0.3);
        legend_handles(i) = scatter3(cx, cy, cz, 120, cmap(i,:), 'filled');
        text(cx, cy, cz, num2str(i), 'FontSize', 12, 'FontWeight', 'bold', 'Color', 'k', 'BackgroundColor', [0.95 0.95 0.95], 'HorizontalAlignment', 'center', 'VerticalAlignment', 'middle');
    end
    legend(legend_handles, conditions, 'Location', 'northeastoutside');
    view(-45, 30);
    xlabel('Canonical axis 1'); ylabel('Canonical axis 2'); zlabel('Canonical axis 3');
    
    for i = 1:3
        nexttile(i*2)
        gl_src_n = (gl_src(i,:) - mean(gl_src(i,:))) / std(gl_src(i,:));
        emb_can_pr_n = (emb_can_pr(i,:) - mean(emb_can_pr(i,:))) / std(emb_can_pr(i,:));
        plot(gl_src_n,'blue'); hold on; plot(emb_can_pr_n,'red');
        title(['comp ', num2str(i), ' | corr = ', num2str(corr(gl_src_n',emb_can_pr_n'),'%.2f')])
        grid(); xticks(ticks(1:end-1)); xlim([0, ticks(end)]);
        conditions_num = arrayfun(@(x) ['(' num2str(x) ')'], 1:22, 'UniformOutput', false);
        xticklabels(conditions_num);
        if i == 1, legend('Global source', 'UMAP projection'); end
        if i == 3, xlabel('Experimental conditions'); end
    end
    xticklabels(conditions);
    exportgraphics(fig4, fullfile(sub_save_dir, '04_Canonical_Structure.png'), 'Resolution', 300);
    
    % =====================================================================
    % PERMUTATION TEST (FIGURE 5) - CONTINUOUS EEG SHIFT
    % =====================================================================
    fprintf('  -> Running permutation test (continuous EEG shift)...\n');
    clear corrmax corrmin
    chs = size(Xfiltpca,1);
    
    % DYNAMIC SIZES TO PREVENT "Index exceeds array bounds"
    samples_per_epoch = size(Epochs_filt, 2);
    nEpochs = size(Epochs_filt, 3);
    
    parfor i=1:1000
        r_idx = fix(rand*size(Xfiltpca,2));
        XCirc = circshift(Xfiltpca,[0,r_idx]);
        
        % Берем точное количество семплов, защищаясь от ошибок длины массива
        total_samples = samples_per_epoch * nEpochs;
        Eps_circ = reshape(XCirc(:, 1:total_samples), chs, samples_per_epoch, nEpochs);
        
        X_test = zeros(size(X_epo));
        en = 0;
        for j=1:nEpochs
            ep_wins = epoch_data(Eps_circ(:,:,j)', Fs, Wsize, Ssize);
            st = en + 1; en = en + size(ep_wins,3);
            X_test(:,:,st:en) = ep_wins; 
        end
        [~,~,~,~,corrs_test] = espoc(X_test, R');
        corrmax(:,i) = max(corrs_test,[],2);
        corrmin(:,i) = min(corrs_test,[],2);
    end
    
    corrmax1 = sort(max(corrmax,[],1),'descend');
    corrmin1 = sort(min(corrmin,[],1));
    alpha = 0.05;
    i=1; while 1 - sum(corrmax1(i) > corrmax1)/numel(corrmax1) <= alpha, i=i+1; end
    max_val = corrmax1(i);
    i=1; while 1 - sum(corrmin1(i) < corrmin1)/numel(corrmin1) <= alpha, i=i+1; end
    min_val = corrmin1(i);
    
    fig5 = figure('Visible', 'off'); 
    stem(corrs'); hold on; yline(max_val); yline(min_val);
    title(sprintf('Permutation thresholds: %s', sub_name), 'Interpreter', 'none');
    exportgraphics(fig5, fullfile(sub_save_dir, '05_Permutation_Test.png'), 'Resolution', 300);
    
    % =====================================================================
    % VISUALIZE ALL FILTERS AND PATTERNS (TOP 5)
    % =====================================================================
    fprintf('  -> Saving TOP components to disk...\n');
    all_comp_dir = fullfile(sub_save_dir, 'All_Components');
    if ~exist(all_comp_dir, 'dir')
        mkdir(all_comp_dir);
    end
    
    n_global = size(W, 1);
    n_local  = size(W, 3);
    n_top_local = min(5, n_local); 
    
    % Переиспользование объекта фигуры экономит ОЗУ.
    fig_comp = figure('Visible', 'off', 'Position', [100, 100, 1600, 400]);
    
    for g_idx = 1:n_global
        gl_dir = fullfile(all_comp_dir, sprintf('Global_UMAP_%02d', g_idx));
        if ~exist(gl_dir, 'dir')
            mkdir(gl_dir);
        end
        
        zz = Vz(:, g_idx)' * Rmean';
        
        % Сортируем локальные компоненты по модулю корреляции
        [~, sorted_l_idx] = sort(abs(corrs(g_idx, :)), 'descend');
        
        for rank_idx = 1:n_top_local
            l_idx = sorted_l_idx(rank_idx);
            
            % ЖЕСТКАЯ ОЧИСТКА ГРАФИКА ОТ ВСЕХ СКРЫТЫХ ОБЪЕКТОВ
            clf(fig_comp, 'reset');
            set(fig_comp, 'Color', 'w');
            
            wx = reshape(W(g_idx, :, l_idx), [], 1);
            ax = reshape(A(g_idx, :, l_idx), [], 1);
            
            ax_sens = U * ax;
            wx_sens = U * wx;
            
            [~, idx_w] = max(abs(wx_sens)); 
            wx_sens = wx_sens .* sign(wx_sens(idx_w));
            [~, idx_a] = max(abs(ax_sens)); 
            ax_sens = ax_sens .* sign(ax_sens(idx_a));
            
            S_dyn = zeros(1, size(Covs, 3));
            for ep = 1:size(Covs, 3)
                S_dyn(ep) = wx' * Covs(:,:,ep) * wx;
            end
            S_dyn = (S_dyn - mean(S_dyn)) / std(S_dyn);
            
            zz_aligned = (zz - mean(zz)) / std(zz) * sign(corrs(g_idx, l_idx));
            
            t_comp = tiledlayout(fig_comp, 1, 4, 'TileSpacing', 'compact', 'Padding', 'compact');
            sgtitle(sprintf('%s | Global %d | Local %d (Rank %d) | eSPoC Corr: %.3f', ...
                sub_name, g_idx, l_idx, rank_idx, corrs(g_idx, l_idx)), 'Interpreter', 'none', 'FontWeight', 'bold');
            
            ax1 = nexttile(t_comp, 1);
            topo.avg = wx_sens; cfg.figure = ax1; ft_topoplotER(cfg, topo); title(ax1, 'Filter');
            
            ax2 = nexttile(t_comp, 2);
            topo.avg = ax_sens; cfg.figure = ax2; ft_topoplotER(cfg, topo); title(ax2, 'Pattern');
            
            ax3 = nexttile(t_comp, 3, [1, 2]);
            plot(ax3, S_dyn, 'LineWidth', 1.2, 'Color', [0 0.4470 0.7410]); hold on;
            plot(ax3, zz_aligned, 'LineWidth', 1.2, 'Color', [0.8500 0.3250 0.0980]);
            grid on;
            
            ticks = 0:tstep:size(S_dyn,2);
            xticks(ax3, ticks(1:end-1));
            xticklabels(ax3, conditions);
            xtickangle(ax3, 45);
            xlim(ax3, [0, ticks(end)]);
            
            for k = 1:length(ticks)-1
                xline(ax3, ticks(k), '--', 'Color', [0.2 0.2 0.2], 'Alpha', 0.5);
            end
            
            legend(ax3, 'Source Envelope', 'UMAP canonical projection', 'Location', 'best');
            
            img_name = sprintf('Rank_%02d_Local_%02d_corr_%.3f.png', rank_idx, l_idx, corrs(g_idx, l_idx));
            exportgraphics(fig_comp, fullfile(gl_dir, img_name), 'Resolution', 150);
        end
    end
    
    % ГАРАНТИЯ ОСВОБОЖДЕНИЯ ПАМЯТИ: закрываем все фигуры испытуемого
    close all; 
end
fprintf('=======================================================\n');
fprintf('ВСЕ ИСПЫТУЕМЫЕ УСПЕШНО ОБРАБОТАНЫ!\n');
fprintf('=======================================================\n');

