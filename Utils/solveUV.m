function  x = solveUV( ktraj, kdata, csm, V, N, nIterations, SBasis, useGPU)
%function  x = solveUV( ktraj, kdata, csm, V, N, nIterations, SBasis, useGPU)
%

[nSamplesPerFrame, numFrames, ~, ~] = size( kdata);
[~, nbasis] = size( V);

if(useGPU)
    osf = 2; wg = 3; sw = 8;
    w = ones( nSamplesPerFrame*numFrames, 1);
    %w=repmat(dcf,[1 5*numFrames]);
    ktraj_gpu = [real( ktraj(:)), imag( ktraj(:))]';
    FT = gpuNUFFT( ktraj_gpu/N, w(:), osf, wg, sw, [N, N], [], true);
    Atb = Atb_UV( FT, kdata, V, csm, N, true);
    Reg = @(x) reshape( reshape( x, [N*N, nbasis]) * SBasis, [N*N*nbasis, 1]);
    AtA = @(x) AtA_UV( FT, x, V, csm, N, nSamplesPerFrame) + Reg( x);
else
    FT = NUFFT( ktraj/N, 1, 0, 0, [N, N]);
    Atb = Atb_UV( FT, kdata, V, csm, false);
    AtA = @(x) AtA_UV( FT, x, V, csm, nSamplesPerFrame);
end

x = pcg_quiet( AtA, Atb(:), 1e-5, nIterations);
x = reshape( x, [N, N, nbasis]);

end

