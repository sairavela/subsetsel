% See previous ChatGPT message for full content
function [Xo, yo, meta] = abess_preprocess(X, y, opts)
% Center columns of X, center y, and (optionally) scale columns to norm sqrt(n),
% matching the paper's assumption Xj'Xj = n.

    if ~isfield(opts,'standardize') || isempty(opts.standardize)
        opts.standardize = true;
    end

    [n, p] = size(X);
    meta = struct();

    if opts.standardize
        muX = mean(X,1);
        Xc  = X - muX;

        muy = mean(y);
        yc  = y - muy;

        colnorm = sqrt(sum(Xc.^2, 1));
        % Scale to have norm sqrt(n): Xj'Xj = n
        scale = colnorm / sqrt(n);
        scale(scale==0) = 1;

        Xo = Xc ./ scale;
        yo = yc;

        meta.muX   = muX;
        meta.scale = scale;
        meta.muy   = muy;
        meta.standardize = true;
    else
        Xo = X;
        yo = y;
        meta.standardize = false;
    end

    meta.n = n;
    meta.p = p;
end
