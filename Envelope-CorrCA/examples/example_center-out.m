close all
clear
clc

ft_path = 'C:\Users\ansbel\Documents\GitHub\CBI\site-packages\fieldtrip\';

if ~exist('ft_defaults','file')
    addpath(ft_path);
end

%% Target epochs
% sub_path = 'D:\OS(CURRENT)\data\parkinson\pathology\Patient_1_CenterOut_OFF_EEG_clean_epochs.fif';
sub_path = 'D:\OS(CURRENT)\data\parkinson\control\Control_7_CenterOut_epochs.fif';
cfg = [];
cfg.dataset = sub_path; 
Epochs_inf = ft_preprocessing(cfg); 

Fs = Epochs_inf.hdr.Fs;

[~, n_ts_ep] = size(Epochs_inf.trial{1});

%%
% idxs = [  0,   1,   2,   3,   6,   8,  12,  13,  15,  16,  17,  19,  22,...
%         24,  25,  28,  30,  32,  33,  35,  37,  39,  41,  42,  44,  45,...
%         48,  50,  51,  55,  57,  58,  59,  62,  63,  65,  69,  70,  73,...
%         74,  75,  78,  79,  82,  85,  88,  89,  92,  93,  94,  97, 100,...
%        101, 103, 105, 106, 110, 111, 112, 117, 118, 121, 122, 124, 126,...
%        128, 131, 132, 134, 136, 137, 138] + 1;
idxs = [  2,   3,   6,   7,   8,  11,  12,  15,  18,  20,  21,  24,  26,...
        28,  30,  32,  34,  37,  38,  40,  42,  43,  46,  48,  50,  52,...
        54,  57,  58,  61,  63,  65,  67,  68,  70,  71,  73,  77,  78,...
        79,  81,  83,  84,  86,  89,  90,  92,  95,  96,  98, 101, 103,...
       105, 106, 108, 109, 113, 115, 118, 119, 121, 123, 126, 127, 128,...
       131, 133, 134, 137, 139, 141, 142, 145, 146] + 1;

all_idxd = 1:numel(Epochs_inf.trial);
% idxs = setdiff(all_idxd, idxs); 

Fmin = 17;
Fmax = 23;
band = [Fmin Fmax];

Wsize = 1/Fmin;
Ssize = Wsize/2;

[b_band,a_band] = butter(4, band/(Fs/2));

Fs = Epochs_inf.hdr.Fs;

clear Epochs Epochs_alg
for ep_idx=1:numel(Epochs_inf.trial)  
    Ep = Epochs_inf.trial{ep_idx}';
    Ep = Ep(:,1:38);
    Epfilt = filtfilt(b_band,a_band,Ep);
    Epochs(:,:,ep_idx) = Epfilt;
    Epochs_alg(:,:,ep_idx) = Epfilt(Fs/2+1:end-Fs/2,:);
end

Epochs = Epochs(:,:,idxs);
Epochs_alg = Epochs_alg(:,:,idxs);

[W, A, z_trials, X_epochs] = env_corrca(Epochs_alg, Fs, Wsize, Ssize);
% [z_trials, W, A, X_epochs, X_covs] = env_grad_dec(Epochs_alg, Fs, Wsize, Ssize);

%%
% z_comp = squeeze(z_trials(:,2,:));
% z_comp = z_comp(:);
% z_comp = repmat(mean(z_trials(:,1,:),3), 1, 72);

% [W,A] = spoc(X_epochs(:,:,:),z_comp(:)');

%%
imagesc(squeeze(z_trials(:,1,:))')
colorbar

%%
plot(mean(z_trials(:,1,:),3))

%%
plot(mean(z_trials(:,2,:),3))

%%
gl_c = 1;
comp_idx = 35;
wx = squeeze(W(gl_c,:,comp_idx))';
patt = squeeze(A(gl_c,:,comp_idx));
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

% Жестко фиксируем границы от -3 до 4
xlim(axH, [-3, 4]);
