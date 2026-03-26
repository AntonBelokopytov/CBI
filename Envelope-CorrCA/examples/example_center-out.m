close all
clear
clc

ft_path = 'D:\OS(CURRENT)\scripts\2Git\fieldtrip\';

if ~exist('ft_defaults','file')
    addpath(ft_path);
end

%% Target epochs
sub_path = 'Patient1_ConditionOff_CenterOut_2-35Hz_clear_epochs.fif';
% sub_path = 'Control_4_CenterOut_epochs.fif';
% sub_path = 'D:/OS(CURRENT)/data/parkinson/control/Control_9_CenterOut_epochs.fif';
cfg = [];
cfg.dataset = sub_path; 
Epochs_inf = ft_preprocessing(cfg); 

Fs = Epochs_inf.hdr.Fs;

[~, n_ts_ep] = size(Epochs_inf.trial{1});

%%
idxs = [  2,   3,   4,   5,   8,  10,  14,  15,  18,  19,  20,  22,  25,...
        27,  28,  31,  33,  35,  36,  38,  41,  43,  45,  46,  48,  49,...
        52,  56,  57,  63,  66,  67,  68,  71,  72,  74,  78,  79,  83,...
        84,  85,  88,  89,  92,  95,  98,  99, 102, 103, 104, 107, 111,...
       112, 114, 116, 117, 122, 123, 124, 130, 131, 135, 136, 138, 141,...
       143, 146, 148, 151, 153, 154, 156] + 1;

Fmin = 8;
Fmax = 12;
band = [Fmin Fmax];

Wsize = 0.1;
Ssize = 0.05;

[b_band,a_band] = butter(4, band/(Fs/2));

Fs = Epochs_inf.hdr.Fs;

clear Epochs
for ep_idx=1:numel(Epochs_inf.trial)  
    Ep = Epochs_inf.trial{ep_idx}';
    Ep = Ep(:,1:38);
    Epfilt = filtfilt(b_band,a_band,Ep);
    Epochs(:,:,ep_idx) = Epfilt;
    Epochs_alg(:,:,ep_idx) = Epfilt(Fs/2+1:end-Fs/2,:);
end
Epochs = Epochs(:,:,idxs);
Epochs_alg = Epochs_alg(:,:,idxs);

n_plot_comp = 4;
% [W, A, eigenvals, Epochs_cov, z] = env_corrca(Epochs_alg, Fs, Wsize, Ssize, n_plot_comp);
% [W, A] = env_pca(Epochs_alg, Fs, Wsize, Ssize);
% [W, A, corrs, z] = env_hilbert_corrca(Epochs_alg,Fs,Wsize,Ssize,n_plot_comp);
% [W, A, eigenvals, Epochs_cov, z] = env_laplace_dec(Epochs_alg, Fs, Wsize, Ssize);
[W, A, eigenvals, Epochs_cov, z] = env_gradient_dec(Epochs_alg, Fs, Wsize, Ssize);

%%
gl_c = 1;
comp_idx = 1;
% wx = squeeze(W(gl_c,:,comp_idx))';
% patt = squeeze(A(gl_c,:,comp_idx));
wx = squeeze(W(:,comp_idx));
patt = squeeze(A(:,comp_idx));

% wx = squeeze(W(:,comp_idx));
% patt = squeeze(A(:,comp_idx));

patt = patt * sign(patt(abs(patt)==max(abs(patt))));

clear Yenv Yseg
for ep_idx=1:size(Epochs,3)
    ep = Epochs(:,:,ep_idx);
    ep = ep / sqrt(trace(cov(ep)));
    en = abs(hilbert(ep*wx));
    Yenv(:,ep_idx) = en;
    Yseg(:,ep_idx) = Epochs(:,:,ep_idx)*wx;
end

Filt = wx;

elec = Epochs_inf.hdr.elec
elec.chanpos  = elec.chanpos(1:38, :);
elec.elecpos  = elec.elecpos(1:38, :);
elec.chantype = elec.chantype(1:38);
elec.chanunit = elec.chanunit(1:38);
elec.label    = elec.label(1:38);

figure; hold on; grid on
set(gcf,'Color','w');

% tsec     = (1:size(Xenv,1))/Fs;
tsec     = linspace(-3, 4, size(Yenv,1));
E        = size(Yenv, 2);
env_mean = mean(Yenv, 2, 'omitnan');
sd       = std (Yenv, 0, 2, 'omitnan');

% Подготовка FieldTrip layout
lay = ft_prepare_layout(struct('elec', elec));

% ==== ГЛАВНАЯ РАСКЛАДКА ====================================================
t = tiledlayout(3,2, 'TileSpacing','compact', 'Padding','compact');

% ---- (Left) Heatmap: все эпохи, огибающая ----
axH = nexttile(t, 1, [3 1]);          % левая колонка, 3 строки
imagesc(axH, tsec, 1:E, Yenv');
set(axH,'YDir','normal','Color','w'); grid(axH,'on');
xline(axH, 0, 'k--', 'LineWidth', 2);
% xline(axH, -2, 'k--', 'LineWidth', 2);
xlabel(axH,'time, s'); ylabel(axH,'epoch');
title(axH,'Envelope per epoch');
colorbar(axH);
caxis(axH, [0 3*max(std(Yenv))]);

% ---- (Right-Top) ERP без усреднения ----
axERP = nexttile(t, 2); hold(axERP,'on'); grid(axERP,'on');
set(axERP,'Color','w');
plot(axERP, tsec, Yseg);
xline(axERP, 0, 'k--', 'LineWidth', 2);
% xline(axERP, -2, 'k--', 'LineWidth', 2);
xlabel(axERP,'time, s'); ylabel(axERP,'amplitude');
title(axERP,'Source activity (all epochs)');

% ---- (Right-Middle) Средняя огибающая ± SD ----
axENV = nexttile(t, 4); hold(axENV,'on'); grid(axENV,'on');
set(axENV,'Color','w');
xfill = [tsec, fliplr(tsec)];
yfill = [ (env_mean+sd).', fliplr((env_mean-sd).') ];
fill(axENV, xfill, yfill, [0.3 0.5 1.0], 'FaceAlpha',0.2, 'EdgeColor','none');
plot(axENV, tsec, env_mean, 'Color', [0.1 0.3 0.9], 'LineWidth', 2);
xline(axENV, 0, 'k--', 'LineWidth', 2);
% xline(axENV, -2, 'k--', 'LineWidth', 2);
xlabel(axENV,'time, s'); ylabel(axENV,'envelope (a.u.)');
title(axENV,'Mean envelope \pm SD');

% ==== (Right-Bottom) ДВА ГРАФИКА: ФИЛЬТР и ПАТТЕРН =========================

% 1) Создаём временную ось в тайле #6, берём её позицию и удаляем
axTmp = nexttile(t, 6);
pos6  = axTmp.OuterPosition;   % можно взять Position, если так удобнее
delete(axTmp);

% 2) На это место ставим панель с белым фоном и вложенный tiledlayout 1×2
figCol = get(gcf,'Color');     % обычно 'w'
p6 = uipanel('Parent', gcf, ...
             'Units','normalized', ...
             'Position', pos6, ...
             'BorderType','none', ...
             'BackgroundColor', figCol);   % белый фон панели

t6 = tiledlayout(p6, 1, 2, 'TileSpacing','compact', 'Padding','compact');

% --------- Левый подтайл: ТОПОГРАФИЯ ФИЛЬТРА ---------
axFiltTopo = nexttile(t6, 1);
set(axFiltTopo, 'Color','w');

topoF = [];
topoF.dimord = 'chan_time';
topoF.label  = elec.label;
topoF.time   = 0;
topoF.avg    = Filt;
topoF.elec   = elec;

cfg = [];
cfg.figure       = axFiltTopo;
cfg.layout       = lay;
cfg.comment      = 'no';
cfg.style        = 'fill';
cfg.markersymbol = 'o';
cfg.zlim         = 'maxmin';

cfg.layout.pos(:, 1:2) = cfg.layout.pos(:, 1:2) * 1.1; 
cfg.layout.pos(:, 2) = cfg.layout.pos(:, 2) - 0.05;

% cfg.colorbar     = 'EastOutside';
ft_topoplotER(cfg, topoF);
title(axFiltTopo,'Filter');

% --------- Правый подтайл: ТОПОГРАФИЯ ПАТТЕРНА ---------
axPatTopo = nexttile(t6, 2);
set(axPatTopo, 'Color','w');

valsPat = patt; valsPat = valsPat(:);
topoP = [];
topoP.dimord = 'chan_time';
topoP.label  = elec.label;
topoP.time   = 0;
topoP.avg    = valsPat;
topoP.elec   = elec;

cfg = [];
cfg.figure       = axPatTopo;
cfg.layout       = lay;
cfg.comment      = 'no';
cfg.style        = 'fill';
cfg.markersymbol = 'o';
cfg.zlim         = 'maxmin';

cfg.layout.pos(:, 1:2) = cfg.layout.pos(:, 1:2) * 1.1; 
cfg.layout.pos(:, 2) = cfg.layout.pos(:, 2) - 0.05;

% cfg.colorbar     = 'EastOutside';
ft_topoplotER(cfg, topoP); 
title(axPatTopo,'Pattern');

% На всякий случай перекрасим все оси внутри панели в белый
set(findall(p6, 'type','axes'), 'Color', figCol);

% ==== Синхронизация осей по X у временных графиков =========================
linkaxes([axH axERP axENV], 'x');

