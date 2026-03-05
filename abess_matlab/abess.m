function out = abess(X, y, opts)
%ABESS  Adaptive Best-Subset Selection via splicing (Zhu et al. 2020).
%
%   out = abess(X, y, opts)
%
% Implements:
%   Algorithm 3 (ABESS): loops over s=1..smax, calls BESS.Fixed(s),
%   selects s minimizing SIC(A_s) = n*log(L_s) + |A_s|*log(p)*log(log(n)).
%
% Inputs
%   X : n-by-p design matrix
%   y : n-by-1 response vector
%   opts fields (optional):
%       .smax        : maximum support size (default floor(n/(log(p)*log(log(n)))))
%       .kmax_rule   : 'min(s,10)' or 's' (default 'min(s,10)')
%       .tau_rule    : 'paper' or numeric scalar (default 'paper')
%       .standardize : true/false (default true)  % center y and columns of X, and scale columns to norm sqrt(n)
%       .maxIter     : max splicing iterations per s (default 200)
%       .ridge       : small ridge added to XtX when ill-conditioned (default 1e-10)
%       .verbose     : 0/1 (default 0)
%
% Output struct out:
%   .beta        : p-by-1 coefficients at selected size
%   .A           : active set indices (1..p)
%   .s           : selected sparsity
%   .SIC_path    : SIC(s) values
%   .L_path      : loss values (Ln) for each s
%   .A_path      : cell array of active sets by s
%   .beta_path   : cell array of betas by s
%   .meta        : preprocessing metadata

    if nargin < 3, opts = struct(); end
    [X, y, meta] = abess_preprocess(X, y, opts);

    [n, p] = size(X);

    if ~isfield(opts, 'smax') || isempty(opts.smax)
        denom = log(max(p,3)) * log(max(log(max(n,3)), 1.000001));
        smax0 = floor(n / max(denom, 1e-6));
        opts.smax = max(1, min(p, smax0));
    else
        opts.smax = max(1, min(p, floor(opts.smax)));
    end

    if ~isfield(opts, 'verbose'), opts.verbose = 0; end

    SIC_path  = inf(opts.smax,1);
    L_path    = inf(opts.smax,1);      % numeric vector
    A_path    = cell(opts.smax,1);     % cell array
    beta_path = cell(opts.smax,1);     % cell array

    for s = 1:opts.smax
        if opts.verbose
            fprintf('[ABESS] s=%d/%d\n', s, opts.smax);
        end
        opts_s = opts;
        opts_s.s = s;
        opts_s.kmax = choose_kmax(s, opts);
        opts_s.tau  = choose_tau(s, n, p, opts);

        res = bess_fixed(X, y, opts_s);

        A_path{s}      = res.A;
        beta_path{s}   = res.beta;
        L_path(s)      = res.L;        % FIX: () not {}
        
        % SIC(A) = n*log(LA) + |A|*log(p)*log(log(n))
        loglogn = log(max(log(max(n,3)), 1.000001));
        SIC_path(s) = n * log(max(res.L, realmin)) + numel(res.A) * log(max(p,3)) * loglogn;
    end

    [~, smin] = min(SIC_path);
    out = struct();
    out.beta      = beta_path{smin};
    out.A         = A_path{smin};
    out.s         = smin;
    out.SIC_path  = SIC_path;
    out.L_path    = L_path;
    out.A_path    = A_path;
    out.beta_path = beta_path;
    out.meta      = meta;
end

function kmax = choose_kmax(s, opts)
    if ~isfield(opts,'kmax_rule') || isempty(opts.kmax_rule)
        opts.kmax_rule = 'min(s,10)';
    end
    switch lower(opts.kmax_rule)
        case 's'
            kmax = s;
        otherwise
            kmax = min(s, 10);
    end
end

function tau = choose_tau(s, n, p, opts)
    if ~isfield(opts,'tau_rule') || isempty(opts.tau_rule)
        opts.tau_rule = 'paper';
    end
    if isnumeric(opts.tau_rule)
        tau = opts.tau_rule;
        return;
    end
    % Suggested in paper: tau_s = 0.01*s*log(p)*log(log(n))/n
    loglogn = log(max(log(max(n,3)), 1.000001));
    tau = 0.01 * s * log(max(p,3)) * loglogn / n;
end