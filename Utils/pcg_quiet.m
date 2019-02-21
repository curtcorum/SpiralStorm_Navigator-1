function [x, flag, relres, iter, resvec] = pcg_quiet( varargin)
%function [x, flag, relres, iter, resvec] = pcg_quiet( varargin)
% quiet version of pcg without text output and with warning
%
% Curt Corum, 2/20/2019
% based on https://stackoverflow.com/questions/21354088/suppress-itermsg-in-matlab

[x, flag, relres, iter, resvec] = pcg( varargin{:});

if flag == 1; warning('pcg_quiet:stopped', 'pcg_quiet stoped at iteration %d with relative residual %d', iter, relres); end
if flag == 2; warning('pcg_quiet:ill_cond', 'pcg_quiet preconditioner is ill conditioned iteration %d and residual %d', iter, relres); end
if flag == 3; warning('pcg_quiet:stagnated', 'pcg_quiet stagnated at iteration %d with relative residual %d', iter, relres); end
if flag == 4; warning('pcg_quiet:over/underflow', 'pcg_quiet over/underflow at iteration %d with relative residual %d', iter, relres); end

end

