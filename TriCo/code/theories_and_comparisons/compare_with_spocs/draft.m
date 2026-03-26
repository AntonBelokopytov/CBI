We(1,:,:) = Wsp1; 
We(2,:,:) = Wsp2;

%%
Env = [];
for i=1:2
    for j=1:64
        for ep_idx=1:size(Epochs_cov,3) 
            w = squeeze(We(i,:,j))';
            Env(i,j,ep_idx) = w' * Epochs_cov(:,:,ep_idx) * w;
        end
    end
end

corr_w1 = corr(z_epo_true', squeeze(Env(1,:,:))')
corr_w2 = corr(z_epo_true', squeeze(Env(2,:,:))')

%%
figure
stem(corr_w1')

figure
stem(corr_w2')

%%
