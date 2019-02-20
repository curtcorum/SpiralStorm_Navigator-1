clear all

addpath( './csm');
addpath( './Utils');
addpath( './nufft_toolbox_cpu');
addpath( genpath( './gpuNUFFT'));
addpath( genpath( './CUDA'));

%% Reconstruction parameters
spiralsToDelete = 60;
framesToDelete = 0;
ninterleavesPerFrame = 6;
N = 340;
nChannelsToChoose = 8;
numFramesToKeep = 500;
useGPU = 'true';
SHRINK_FACTOR = 1.0;
nBasis = 30;
lambdaSmoothness = 0.025;
cRatioI = 1:nChannelsToChoose;
sigma = [4.5];
lam = [0.1];
%%
% % ==============================================================
% % Load the data
% % ============================================================== 
load( 'Series8.mat'); 
%kdata = kdata(:,:,:,4); % Fourth Slice Data


 %% =========================================
 % -------------Preprocessing Data-------------%
 %===========================================
[nFreqEncoding, nCh,numberSpirals] = size( kdata);
numFrames = floor( (numberSpirals-spiralsToDelete)/ninterleavesPerFrame);
kdata = kdata(:, cRatioI(1:nChannelsToChoose), spiralsToDelete + 1:numberSpirals);
k = k(:, spiralsToDelete + 1:numberSpirals);
w = dcf(:, spiralsToDelete + 1:numberSpirals);
kdata = kdata(:, :, 1:numFrames*ninterleavesPerFrame);
k = k(:, 1:numFrames*ninterleavesPerFrame);
w = w(:, 1:numFrames*ninterleavesPerFrame);

kdata = permute( kdata, [1,3,2]);
kdata = reshape( kdata, [nFreqEncoding, ninterleavesPerFrame, numFrames, nChannelsToChoose]);
ktraj = k;
ktraj = reshape( ktraj, [nFreqEncoding, ninterleavesPerFrame, numFrames]);
w = reshape( w, [nFreqEncoding, ninterleavesPerFrame, numFrames]);

% Keeping only numFramesToKeep

kdata = kdata(:, :, framesToDelete + 1:numFramesToKeep + framesToDelete, cRatioI(1:nChannelsToChoose));
ktraj = ktraj(:, :, framesToDelete + 1:numFramesToKeep + framesToDelete);
%save data kdata ktraj dcf

%% ==============================================================
% Scaling trajectory
% ==============================================================

ktraj_scaled =  SHRINK_FACTOR*ktraj*N;

%% ==============================================================
% Compute the coil sensitivity map
% ============================================================== 
ktraj_scaled = reshape( ktraj_scaled, [nFreqEncoding, ninterleavesPerFrame, numFramesToKeep]);
kdata = reshape( kdata, [nFreqEncoding, ninterleavesPerFrame, numFramesToKeep, nChannelsToChoose]);
[coilImages] = coil_sens_map_NUFFT( kdata, ktraj_scaled, N, useGPU);

%% ===============================================================
% Compute coil compresession
% ================================================================
kdata = reshape( kdata, [nFreqEncoding*ninterleavesPerFrame*numFramesToKeep, nChannelsToChoose]);

[vkdata, vcoilImages] = combine_coils( kdata, coilImages, 0.85); % 0.85 parameter in variable, *** CAC 190220 
nChannelsToChoose = size( vcoilImages, 3);
kdata = reshape( vkdata, [nFreqEncoding, ninterleavesPerFrame, numFramesToKeep, nChannelsToChoose]);
csm = giveEspiritMaps( reshape( vcoilImages, [size( vcoilImages, 1), size( vcoilImages, 2), nChannelsToChoose]));
coilImages = vcoilImages;

ktraj_scaled = reshape( ktraj_scaled, [nFreqEncoding, ninterleavesPerFrame, numFramesToKeep]);
kdata = reshape( kdata, [nFreqEncoding, ninterleavesPerFrame, numFramesToKeep, nChannelsToChoose]);

%% ==============================================================
% % Compute the weight matrix
% % ============================================================= 
no_ch = size( csm, 3);
Nav = permute( kdata(:, 1, :, :), [1, 2, 4, 3]);

for ii = 1:size( sigma, 2)
    for jj = 1:size( lam, 2)
[~, ~, L] = estimateLapKernelLR( reshape( Nav, [nFreqEncoding*no_ch, numFramesToKeep]), sigma(ii), lam(jj));
[~, Sbasis, V] = svd( L);
V = V(:, end - nBasis + 1:end);
Sbasis = Sbasis(end - nBasis + 1:end, end - nBasis + 1:end);

%% ==============================================================
% % Final Reconstruction
% % ============================================================= 
ktraj_scaled = reshape( ktraj_scaled,[nFreqEncoding*ninterleavesPerFrame,numFramesToKeep]);
kdata = reshape( kdata, [nFreqEncoding*ninterleavesPerFrame, numFramesToKeep, nChannelsToChoose]);
tic;
x = solveUV( ktraj_scaled, kdata, csm, V, N, 60, lambdaSmoothness*Sbasis, useGPU);
toc
y = reshape( reshape( x, [N*N, nBasis]) * V', [N, N, numFramesToKeep]);

%% ==============================================================
% % Save and Display results
% % ============================================================= 

%for i=1:530;imagesc(fliplr(flipud(abs(y(:,:,i))))); pause(0.1); colormap gray;end
%for i=1:250;imagesc(((abs(y(:,:,i))))); pause(0.1); colormap gray; end
%clear kdata csm V ;

% bug, velow commented out, what is var 'sl', CAC 190219 
%save(strcat('res_iter_',num2str(lambdaSmoothness),'_',num2str(sigma(ii)),'_',num2str(sl),'.mat'),'y','-v7.3');

%cd './../../../SpiralcodeUVv2/SpiralcodeUV_new';
    end
end

%% movie display, CAC 190219
sy = size(y);
for idx_t = 1:sy(3)
    colormap gray;
    imagesc( abs( y(:, :, idx_t)));
    Mv(idx_t) = getframe;
end
movie( Mv);

%% save everything, CAC 190220
save( 'recon_cp_to_new_name');  % add a date/time stamp to name, *** CAC 190220

