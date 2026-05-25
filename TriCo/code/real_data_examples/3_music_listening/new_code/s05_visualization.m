% Предварительные расчеты для кластеров 3D графиков
x = Vz(:,1)' * Rmean'; y = Vz(:,2)' * Rmean'; z = Vz(:,3)' * Rmean';
cmap = jet(nEpochs);
ccx=[]; ccy=[]; ccz=[]; mask = 1:N_epoch_trial;

for i = 1:nEpochs
    if mask(end) <= numel(x)
        sc_x = x(mask); sc_y = y(mask); sc_z = z(mask);
    else
        sc_x = x(mask(1):end); sc_y = y(mask(1):end); sc_z = z(mask(1):end);
    end
    mask = mask + N_epoch_trial;
    ccx = [ccx, mean(sc_x)]; ccy = [ccy, mean(sc_y)]; ccz = [ccz, mean(sc_z)];
end

%% =====================================================================
% FIGURE 1: VISUALIZE UMAP 3D SCATTER
% =====================================================================
figure; set(gcf, 'Color', 'w');
scatter3(R(:,1),R(:,2),R(:,3));
xlabel('UMAP component 1'); ylabel('UMAP component 2'); zlabel('UMAP component 3');
title('UMAP Embedding');

%% =====================================================================
% FIGURE 2: Temporal evolution of UMAP components
% =====================================================================
figure; set(gcf, 'Color', 'w');
plot(R)                          
xticks(ticks(1:end-1)); xlim([0, size(R,1)]);
ylabel('UMAP component coordinate'); xlabel('Experimental conditions (Transitions)');
legend({'component 1', 'component 2', 'component 3'})
title('Temporal evolution of UMAP components');

%% =====================================================================
% FIGURE 3: Plot correlation values
% =====================================================================
figure; set(gcf,'Color','w');
stem(corrs','LineWidth',1.5); grid on;
xlabel('Local component index'); ylabel('Correlation');
title('eSPoC correlation values');
legend({'Global 1','Global 2','Global 3'}, 'Location','best');
xlim([1 size(corrs,2)]);

%% =====================================================================
% FIGURE 4: Canonical projections and cluster structure (Tiled Layout)
% =====================================================================
figure; set(gcf,'Color','w');
tiledlayout(3,2)

% --- Left side: 3D cluster ---
nexttile(1,[3,1])
plot3(ccx, ccy, ccz, 'k', 'LineWidth', 1); hold on; grid on
legend_handles = gobjects(nEpochs,1);
mask = 1:N_epoch_trial;
for i = 1:nEpochs
    if mask(end) <= numel(x)
        sc_x = x(mask); sc_y = y(mask); sc_z = z(mask);
    else
        sc_x = x(mask(1):end); sc_y = y(mask(1):end); sc_z = z(mask(1):end);
    end
    mask = mask + N_epoch_trial;
    cx = mean(sc_x); cy = mean(sc_y); cz = mean(sc_z);
    
    scatter3(sc_x, sc_y, sc_z, 10, repmat(cmap(i,:), length(sc_x), 1), 'filled', 'MarkerFaceAlpha', 0.3);
    legend_handles(i) = scatter3(cx, cy, cz, 120, cmap(i,:), 'filled');
    text(cx, cy, cz, num2str(i), 'FontSize', 16, 'FontWeight', 'bold', 'Color', 'k', ...
        'BackgroundColor', [0.95 0.95 0.95], 'HorizontalAlignment', 'center', 'VerticalAlignment', 'middle');
end
legend(legend_handles, conditions, 'Location', 'northeastoutside'); view(-45, 30);
xlabel('Canonical axis 1'); ylabel('Canonical axis 2'); zlabel('Canonical axis 3');

% --- Right side: Temporal dynamics ---
for i = 1:3
    nexttile(i*2)
    gl_src_n = (gl_src(i,:) - mean(gl_src(i,:))) / std(gl_src(i,:));
    emb_can_pr_n = (emb_can_pr(i,:) - mean(emb_can_pr(i,:))) / std(emb_can_pr(i,:));
    plot(gl_src_n,'blue'); hold on; plot(emb_can_pr_n,'red');
    title(['component ', num2str(i), ' | corr = ', num2str(corr(gl_src_n',emb_can_pr_n'),'%.2f')])
    grid on; xticks(ticks(1:end-1)); xlim([0, ticks(end)]);
    conditions_num = arrayfun(@(val) ['(' num2str(val) ')'], 1:nEpochs, 'UniformOutput', false);
    xticklabels(conditions_num);
    if i == 1, legend('Global source signal', 'UMAP canonical projection'); end
    if i == 3, xlabel('Experimental conditions'); end
end
xticklabels(conditions);

%% =====================================================================
% FIGURE 5: SIGNIFICANCE THRESHOLDS
% =====================================================================
figure; set(gcf, 'Color', 'w'); 
stem(corrs'); hold on
yline(max_val, 'r', 'Alpha 0.05 Max'); yline(min_val, 'b', 'Alpha 0.05 Min');
title('Significance Thresholds');

%% =====================================================================
% FIGURE 6: Standalone Canonical space visualization
% =====================================================================
figure; set(gcf,'Color','w');
plot3(ccx, ccy, ccz, 'k', 'LineWidth', 1); hold on; grid on
legend_handles2 = gobjects(nEpochs,1);
mask = 1:N_epoch_trial;
for i = 1:nEpochs
    if mask(end) <= numel(x)
        sc_x = x(mask); sc_y = y(mask); sc_z = z(mask);
    else
        sc_x = x(mask(1):end); sc_y = y(mask(1):end); sc_z = z(mask(1):end);
    end
    mask = mask + N_epoch_trial;
    cx = mean(sc_x); cy = mean(sc_y); cz = mean(sc_z);
    
    scatter3(sc_x, sc_y, sc_z, 10, repmat(cmap(i,:), length(sc_x), 1), 'filled', 'MarkerFaceAlpha', 0.3); hold on
    legend_handles2(i) = scatter3(cx, cy, cz, 120, cmap(i,:), 'filled');
    text(cx, cy, cz, num2str(i), 'FontSize', 16, 'FontWeight', 'bold', 'Color', 'k', ...
        'BackgroundColor', [0.95 0.95 0.95], 'Margin', 0.00001, 'HorizontalAlignment', 'center', 'VerticalAlignment', 'middle');
end
legend(legend_handles2, conditions, 'Location', 'northeastoutside'); view(-45, 30);
xlabel('Canonical axis 1'); ylabel('Canonical axis 2'); zlabel('Canonical axis 3');

%% =====================================================================
% FIGURE 7: 2x2 Layout (Single Global & Local component)
% =====================================================================
gl_src_idx  = 2;
lcl_src_idx = 1;
ax = U*A(gl_src_idx,:,lcl_src_idx)';
wx = U*W(gl_src_idx,:,lcl_src_idx)';
[~, idx] = max(abs(wx)); wx = wx.*sign(wx(idx));
[~, idx] = max(abs(ax)); ax = ax.*sign(ax(idx));

figure; set(gcf,'Color','w');
t = tiledlayout(2,2,'TileSpacing','compact','Padding','compact');
sgtitle(['Source Envelope - UMAP correlation: ', num2str(corrs(gl_src_idx,lcl_src_idx))])

ax1 = nexttile(t,1); title(ax1,'Filter');
topo.avg = wx; cfg_topo.figure = ax1; ft_topoplotER(cfg_topo, topo); 
ax2 = nexttile(t,2); title(ax2,'Pattern');
topo.avg = ax; cfg_topo.figure = ax2; ft_topoplotER(cfg_topo, topo); 

ax3 = nexttile(t,3,[1,2]); hold on; grid on        
title(ax3,'Latent Source Signal & UMAP canonical projection');

S_plot = zeros(1, size(valid_Covs,3));
for i = 1:size(valid_Covs,3)
    S_plot(i) = wx' * U * valid_Covs(:,:,i) * U' * wx;
end
S_plot = (S_plot-mean(S_plot))/std(S_plot);

zz = Vz(:,gl_src_idx)'*Rmean';
zz = (zz-mean(zz))/std(zz) * sign(corrs(gl_src_idx,lcl_src_idx));

plot(S_plot,'LineWidth',1); plot(zz,'LineWidth',1,'Color','red')
xticks(ticks(1:end-1)); xlim([0 ticks(end)])
xticklabels(conditions); xtickangle(45);
xlabel('Experimental conditions'); ylabel('Source signal')
legend('Source Signal Envelope','UMAP canonical projection')

%% =====================================================================
% FIGURE 8: 4x5 Layout (Multiple Components)
% =====================================================================
figure; set(gcf, 'Color', 'w');
t2 = tiledlayout(4,5,'TileSpacing','compact','Padding','compact');
comp_indices = [2 3 4 5];  
all_right_axes = [];       

for i = 1:length(comp_indices)
    lcl_src_idx_mult = comp_indices(i);   
    ax = U*A(gl_src_idx,:,lcl_src_idx_mult)';
    wx = U*W(gl_src_idx,:,lcl_src_idx_mult)';
    
    [~, idx] = max(abs(wx)); wx = wx .* sign(wx(idx));
    [~, idx] = max(abs(ax)); ax = ax .* sign(ax(idx));
    row = (i-1)*5;
    
    ax_wx = nexttile(t2, row + 1);
    topo.avg = wx; cfg_topo.figure = ax_wx; ft_topoplotER(cfg_topo, topo);
    if i==1, title('Filter'); end
    
    ax_ax = nexttile(t2, row + 2);
    topo.avg = ax; cfg_topo.figure = ax_ax; ft_topoplotER(cfg_topo, topo);
    if i==1, title('Pattern'); end
    
    ax_plot = nexttile(t2, row + 3, [1 3]);
    all_right_axes = [all_right_axes ax_plot];
    
    S_plot = zeros(1, size(valid_Covs,3));
    for j = 1:size(valid_Covs,3)
        S_plot(j) = wx' * U * valid_Covs(:,:,j) * U' * wx;
    end
    S_plot = (S_plot - mean(S_plot)) / std(S_plot);
    
    plot(ax_plot, S_plot, 'LineWidth', 1); hold on;
    plot(ax_plot, zz, 'LineWidth', 1, 'Color', 'red');
    grid on
    
    xticks(ax_plot, ticks(1:end-1)); xticklabels(ax_plot, []); xlim(ax_plot, [0, ticks(end)])
    title(ax_plot, ['corr = ', num2str(corrs(gl_src_idx,lcl_src_idx_mult),'%.2f')])
end

conditions_num = arrayfun(@(val) ['(' num2str(val) ')'], 1:nEpochs, 'UniformOutput', false);
for i = 1:3
    ax_dyn = all_right_axes(i);
    xticklabels(ax_dyn, conditions_num);
end
ax_last = all_right_axes(end);
xticklabels(ax_last, conditions); xtickangle(ax_last, 45);
xlabel(ax_last, 'Experimental conditions')

for i = 1:4
    ax_dyn = all_right_axes(i);
    for k = 1:length(ticks(2:end-1))
        xline(ax_dyn, ticks(k+1), '--', 'Color', [0.1 0.1 0.1]);
    end
end
legend(all_right_axes(1), 'Source Signal Envelope', 'UMAP canonical projection');