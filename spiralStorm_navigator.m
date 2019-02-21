%spiralStorm_navigator.m
%
% Ahmed, Abdul Haseeb <abdul-ahmed@uiowa.edu>
%
% beginnings of code cleanup, CAC 190219

%% Environment setup, parameters, logging and debugging
clear all;      % warn about this first, *** CAC 190220

ss_nav_version = '190220.01';  % version
runtime_str = datestr(now, 'yymmdd_HHMMSS');
startPath = pwd;

FLAGS.DEBUG = 2;    %0 off, 1 minimal, 2 log , 3 more, 4 lots...
FLAGS.WARNING = 1;  %0 off, 1 on

if FLAGS.WARNING >= 1; warning('ON'); else warning('OFF'); end

if FLAGS.DEBUG >= 1; tstart = tic; end
if FLAGS.DEBUG >= 2; diary_file = strcat( 'spiralstorm_nav_', runtime_str, '.log'); diary( diary_file); end
if FLAGS.DEBUG >= 1; fprintf( '===== spiralStorm_navigator version: %s =====\n', ss_nav_version);  end
if FLAGS.DEBUG >= 2; fprintf( '%s\n', runtime_str); end

if FLAGS.DEBUG >= 3
    ver;
    fprintf( '==================\n');
    fprintf( 'Git information...\n');
    system('git remote -v; git branch -v --no-abbrev; git status --porcelain;');
    fprintf( '==================\n');
end
% add diff for DEBUG >= 4, *** CAC 190221

if FLAGS.DEBUG >= 1; fprintf( 'clearing variables, saving matlabpath, updating path...\n');  end

%save and setup paths
mlp = path;
restoredefaultpath;
addpath( './csm');
addpath( './Utils');
%addpath( './nufft_toolbox_cpu');
%addpath( genpath( './gpuNUFFT'));
%addpath( genpath( './CUDA'));

if FLAGS.DEBUG >= 4; path,  end

% Reconstruction parameters
spiralsToDelete = 60;
framesToDelete = 0;
ninterleavesPerFrame = 6;
N = 340;                % reconstruction matrix size
nChannelsToChoose = 8;  % starting number of virtual coils
numFramesToKeep = 122;  %numFramesToKeep = 500;
useGPU = 'true';        % 'false' not working yet
SHRINK_FACTOR = 1.0;
nBasis = 30;
lambdaSmoothness = 0.025;
cRatioI = 1:nChannelsToChoose;
sigma = [4.5];          % tuning parameter
lam = [0.1];            % tuning parameter

% new parameters
nIterations = 20;       %nIterations = 60;      % iterations for final reconstuction
nIterations_csm = 20;   %nIterations_csm = 70;  % iterations for coil sensitivity map
eigThresh_1 = 0.02;     %eigThresh_1 = 0.008:   % threshold for picking singular vercors of the calibration matrix (relative to largest singlular value.)
eigThresh_2 = 0.95;     %eigThresh_2 = 0.95;    % threshold of eigen vector decomposition in image space.
%%
% % ==============================================================
% % Load the data
% % ==============================================================
if FLAGS.DEBUG >= 1; tload = tic; end
if FLAGS.DEBUG >= 1; fprintf( 'loading data...');  end

load( 'Series8.mat'); 
kdata = kdata(:,:,:,4); % Fourth Slice Data

if FLAGS.DEBUG >= 1; toc( tload), end

 %% =========================================
 % -------------Preprocessing Data-------------%
 %===========================================
[nFreqEncoding, nCh, numberSpirals] = size( kdata);
numFrames = floor( (numberSpirals-spiralsToDelete)/ninterleavesPerFrame);
kdata = kdata(:, cRatioI(1:nChannelsToChoose), spiralsToDelete + 1:numberSpirals);
%numberSpirals
%size( k)
k = k(:, spiralsToDelete+1:numberSpirals);
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
if FLAGS.DEBUG >= 1; tcsm = tic; end
if FLAGS.DEBUG >= 1; fprintf( 'computing coil sensitivity map for %d channels...', nChannelsToChoose);  end

ktraj_scaled = reshape( ktraj_scaled, [nFreqEncoding, ninterleavesPerFrame, numFramesToKeep]);
kdata = reshape( kdata, [nFreqEncoding, ninterleavesPerFrame, numFramesToKeep, nChannelsToChoose]);
[coilImages] = coil_sens_map_NUFFT( kdata, ktraj_scaled, N, useGPU, nIterations_csm);

if FLAGS.DEBUG >= 1; toc( tcsm), end

%% ===============================================================
% Compute coil compresession
% ================================================================
if FLAGS.DEBUG >= 1; tccc = tic; end
if FLAGS.DEBUG >= 1; fprintf( 'computing coil compression...');  end

kdata = reshape( kdata, [nFreqEncoding*ninterleavesPerFrame*numFramesToKeep, nChannelsToChoose]);

[vkdata, vcoilImages] = combine_coils( kdata, coilImages, 0.85); % 0.85 parameter in variable, *** CAC 190220 
nChannelsToChoose = size( vcoilImages, 3);
kdata = reshape( vkdata, [nFreqEncoding, ninterleavesPerFrame, numFramesToKeep, nChannelsToChoose]);
csm = giveEspiritMaps( reshape( vcoilImages, [size( vcoilImages, 1), size( vcoilImages, 2), nChannelsToChoose]), eigThresh_1, eigThresh_2);
coilImages = vcoilImages;

ktraj_scaled = reshape( ktraj_scaled, [nFreqEncoding, ninterleavesPerFrame, numFramesToKeep]);
kdata = reshape( kdata, [nFreqEncoding, ninterleavesPerFrame, numFramesToKeep, nChannelsToChoose]);

if FLAGS.DEBUG >= 1; toc( tccc), end

%% ==============================================================
% % Compute the weight matrix
% % =============================================================
if FLAGS.DEBUG >= 1; tcwm = tic; end
if FLAGS.DEBUG >= 1; fprintf( 'computing weight matrix...');  end

no_ch = size( csm, 3);
Nav = permute( kdata(:, 1, :, :), [1, 2, 4, 3]);

%uncomment for tuning matrix of recons
%
% ss2 = size( sigma, 2)
% sl2 = size( lam, 2)
% for ii = 1:size( sigma, 2)
%     for jj = 1:size( lam, 2)
%[~, ~, L] = estimateLapKernelLR( reshape( Nav, [nFreqEncoding*no_ch, numFramesToKeep]), sigma(ii), lam(jj));

[~, ~, L] = estimateLapKernelLR( reshape( Nav, [nFreqEncoding*no_ch, numFramesToKeep]), sigma(1), lam(1));

[~, Sbasis, V] = svd( L);
V = V(:, end - nBasis + 1:end);
Sbasis = Sbasis(end - nBasis + 1:end, end - nBasis + 1:end);

if FLAGS.DEBUG >= 1; toc( tcwm), end

%% ==============================================================
% % Final Reconstruction
% % ============================================================= 
if FLAGS.DEBUG >= 1; tfr = tic; end
if FLAGS.DEBUG >= 1; fprintf( 'final reconstruction...');  end

ktraj_scaled = reshape( ktraj_scaled, [nFreqEncoding*ninterleavesPerFrame, numFramesToKeep]);
kdata = reshape( kdata, [nFreqEncoding*ninterleavesPerFrame, numFramesToKeep, nChannelsToChoose]);

x = solveUV( ktraj_scaled, kdata, csm, V, N, nIterations, lambdaSmoothness*Sbasis, useGPU); % nIterations now variable at top, CAC 190220

y = reshape( reshape( x, [N*N, nBasis]) * V', [N, N, numFramesToKeep]);

if FLAGS.DEBUG >= 1; toc( tfr), end

%% ==============================================================
% % Save and Display results
% % ============================================================= 

%for i=1:530;imagesc(fliplr(flipud(abs(y(:,:,i))))); pause(0.1); colormap gray;end
%for i=1:250;imagesc(((abs(y(:,:,i))))); pause(0.1); colormap gray; end
%clear kdata csm V ;

% bug, velow commented out, what is var 'sl', CAC 190219 
%save(strcat('res_iter_',num2str(lambdaSmoothness),'_',num2str(sigma(ii)),'_',num2str(sl),'.mat'),'y','-v7.3');

%cd './../../../SpiralcodeUVv2/SpiralcodeUV_new';
%     end
% end

%% movie display, CAC 190219
if FLAGS.DEBUG >= 1; tmv = tic; end
if FLAGS.DEBUG >= 1; fprintf( 'making movie...');  end

sy = size(y);
for idx_t = 1:sy(3)
    colormap gray;
    imagesc( abs( y(:, :, idx_t)));
    Mv(idx_t) = getframe;
end

movie( Mv, 999);

% save of .mov file or other would go here, *** CAC 190221

if FLAGS.DEBUG >= 1; toc( tmv), end

%% save everything, CAC 190220
if FLAGS.DEBUG >= 1; tsave = tic; end
if FLAGS.DEBUG >= 1; fprintf( 'saving everything in: spiralstorm_nav_recon_please_rename.mat...');  end

save_file = strcat( 'spiralstorm_nav_', runtime_str);
save( save_file);

if FLAGS.DEBUG >= 1; toc( tsave), end

%% restore environment
path( mlp);

%% total elapsed time
if FLAGS.DEBUG >= 1; fprintf( 'Total ');toc( tstart), end
if FLAGS.DEBUG >= 2; diary off; end


