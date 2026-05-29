%% MAIN PIPELINE
close all; clear; clc;

%%
disp('--- 1. Настройка и загрузка данных ---');
run('s01_setup.m');

%%
disp('--- 2. Препроцессинг, SVD и эпохирование ---');
run('s02_preprocessing.m');

%%
disp('--- 3. eSPoC, UMAP и статистика ---');
run('s03_umap_espoc.m');

%%
disp('--- 3. eSPoC, UMAP и статистика ---');
run('s04_stats.m');

%%
disp('--- 4. Отрисовка результатов ---');
run('s05_visualization.m');

%%
