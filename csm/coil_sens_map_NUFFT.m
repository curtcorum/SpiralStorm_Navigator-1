function [csm, coilimages] = coil_sens_map_NUFFT( kdata, trajectory, N, useGPU, nIterations_csm)
%function [csm, coilimages] = coil_sens_map_NUFFT( kdata, trajectory, N, useGPU, nIterations_csm )
%

% use the varargin scheme here instead for backwards compatibility? *** CAC 190220

[nReadouts, nInterleaves, nFrames, nCh] = size( kdata); 

coilimages = zeros( N, N, nCh);

if(useGPU)
    osf = 2; wg = 3; sw = 8;
    %w=ones(nReadouts*nInterleaves*nFrames,1);
    %w=repmat(dcf,[1 nInterleaves*nFrames]);
    ktraj_gpu = [real( trajectory(:)), imag( trajectory(:))]';
    FT = gpuNUFFT( ktraj_gpu/N, ones( nReadouts*nInterleaves*nFrames, 1), osf, wg, sw, [N, N], [], true);
else
    %w=repmat(dcf,[1 nInterleaves]);
    FT = NUFFT( trajectory(:)/N, ones( nReadouts*nInterleaves*nFrames, 1), 0, 0, [N, N]);   % make wrapper for gpuNUFFT and NUFFT with useGPU as argument? *** CAC 190220 
    %Atb = Atb_UV(FT,kdata,V,csm,false);
    %AtA = @(x) AtA_UV(FT,x,V,csm,nFreqEncoding*ninterleavesPerFrame);
end

ATA = @(x) reshape( FT'*(FT*reshape( x, [N, N, 1])), [N*N, 1]);

kdata = reshape( kdata, [nReadouts*nInterleaves*nFrames, nCh]);

for i = 1:nCh
    temp = FT'*kdata(:, i);
    %coilimages(:, :, i) = reshape( pcg_quiet( ATA, temp(:), 1e-6, 70), [N, N]);
    coilimages(:, :, i) = reshape( pcg_quiet( ATA, temp(:), 1e-6, nIterations_csm), [N, N]);  % also pull out threshold in argument/variable, *** CAC 190220
end

csm = giveEspiritMaps( reshape( coilimages, [size( coilimages, 1), size( coilimages, 2), nCh]));

end