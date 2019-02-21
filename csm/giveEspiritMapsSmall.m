function maps = giveEspiritMapsSmall( coilimages, sx, sy, varargin)
%function maps = giveEspiritMapsSmall( coilimages, sx, sy, varargin)
%function maps = giveEspiritMapsSmall( coilimages, sx, sy, eigThresh_1, eigThresh_2)
%
% eigThresh_1 = 0.02 (default) for calibration matrix
%
% eigThresh_2 = 0.95 (default) for calibration matrix
%
% CAC 190220 - varargin added for updating eigThresh

eigThresh_1 = 0.02; % Threshold for picking singular vercors of the calibration matrix (relative to largest singlular value.)
eigThresh_2 = 0.95; % threshold of eigen vector decomposition in image space.
if nargin > 3; eigThresh_1 = varargin{1}; end
if nargin > 4; eigThresh_2 = varargin{2}; end
if nargin > 5; warning('too many arguments'); end

coilimages = fftshift( fftshift( coilimages, 1), 2);
DATA = fftshift( fftshift( fft2( coilimages), 1), 2);

[~, ~, Nc] = size( DATA);
ncalib = 64; 
ksize = [4, 4]; 

% crop a calibration area
calib = crop( DATA, [ncalib, ncalib], Nc);


%% Compute ESPIRiT EigenVectors
% Here we perform calibration in k-space followed by an eigen-decomposition
% in image space to produce the EigenMaps. 

% compute Calibration matrix, perform 1st SVD and convert singular vectors
% into k-space kernels

[k, S] = dat2Kernel( calib, ksize);
idx = max( find( S >= S(1)*eigThresh_1));


%%
% crop kernels and compute eigen-value decomposition in image space to get
% maps
[M, W] = kernelEig( k(:, :, :, 1:idx), [sx, sy]);

% crop sensitivity maps 
maps = M(:, :, :, end).*repmat( W(:, :, end) > eigThresh_2, [1, 1, Nc]);
