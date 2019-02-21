function data = l2Recon(kdata,ktraj,coilImages,lambda,dcf)
%UNTITLED4 Summary of this function goes here
%   Detailed explanation goes here

N = ceil(max([max(abs(imag(ktraj(:)))),max(abs(imag(ktraj(:))))]));
csm = giveEspiritMapsSmall(coilImages,N,N);
[~,~,nFrames,nCh] = size(kdata); 
data = zeros([N,N,nFrames]);
[nSamples,nInterleavesPerFrame,~] = size(kdata);

ktraj = reshape(ktraj/N,[nSamples,nInterleavesPerFrame,nFrames]);
kdata = reshape(kdata, [nSamples,nInterleavesPerFrame,nFrames,nCh]);
w=repmat(dcf(1:nSamples),[1 nInterleavesPerFrame]);
FT= NUFFT(ktraj,sqrt(w(:)),0,0,[N,N]);

Atb = zeros(N,N,nFrames);
for ii=1:nCh
    Atb = Atb + bsxfun(@times,FT'*kdata(:,:,:,ii),conj(csm(:,:,ii)));
end

[~,~,L]=estimateLapKernelLR(reshape(Atb,[size(Atb,1)*size(Atb,2),size(Atb,3)]),1e-5,1);
%ATA = @(x) SenseATA(x,FT,csm,N,nFrames,nCh) + lambda*x;
ATA = @(x) SenseATA(x,FT,csm,N,nFrames,nCh) + lambda*XL(x,L,N,nFrames);
data = pcg_quiet(ATA,Atb(:),1e-5,30);
data = reshape(data,[N,N,nFrames]);

end

function res=XL(x,L,N,nFrames)
x=reshape(x,[N*N nFrames]);
res=x*L;
res=res(:);
end
