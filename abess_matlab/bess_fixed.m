function res = bess_fixed(X, y, opts)
%BESS_FIXED  Best subset selection with fixed support size s (Algorithm 1).
%
% Initialization: pick s variables with largest |corr(X_j,y)| using
% |X_j' y| / sqrt(X_j'X_j) (paper eq in Alg 1 init).
% Then iterate Splicing until active set stable (or maxIter hit).

    [n, p] = size(X);
    if ~isfield(opts,'s'), error('opts.s required'); end
    s = opts.s;
    if ~isfield(opts,'kmax') || isempty(opts.kmax), opts.kmax = min(s,10); end
    if ~isfield(opts,'tau')  || isempty(opts.tau),  opts.tau  = 0; end
    if ~isfield(opts,'maxIter') || isempty(opts.maxIter), opts.maxIter = 200; end
    if ~isfield(opts,'ridge') || isempty(opts.ridge), opts.ridge = 1e-10; end

    % ---- Step 2: Initialize A0 by correlation magnitude (Alg 1).
    xTy = X' * y;
    xTx_diag = sum(X.^2, 1)';   % p-by-1, equals n after normalization
    score = abs(xTy) ./ sqrt(max(xTx_diag, realmin));
    [~, idx] = sort(score, 'descend');
    A = sort(idx(1:s));
    I = setdiff(1:p, A);

    % Compute initial beta and d (Alg 1)
    beta = zeros(p,1);
    betaA = solve_ls(X(:,A), y, opts.ridge);
    beta(A) = betaA;

    r = y - X*beta;
    d = (X' * r) / n;
    d(A) = 0;  % as in Alg 1 init

    % Iterate splicing until convergence
    for m = 1:opts.maxIter
        [beta_new, d_new, A_new, I_new, L_new, improved] = ...
            splicing_step(X, y, beta, d, A, I, opts.kmax, opts.tau, opts.ridge);

        if isequal(A_new, A)
            beta = beta_new; d = d_new; A = A_new; I = I_new;
            break;
        end
        beta = beta_new; d = d_new; A = A_new; I = I_new;

        if ~improved
            % splicing_step may already keep same (A,I) if no significant improvement
            break;
        end
    end

    % Final loss
    L = 0.5/n * (norm(y - X*beta)^2);

    res = struct();
    res.beta = beta;
    res.d    = d;
    res.A    = A;
    res.I    = I;
    res.L    = L;
end

function betaA = solve_ls(XA, y, ridge)
    % Solve (XA'XA) beta = XA'y with tiny ridge for stability
    G = XA' * XA;
    if ridge > 0
        G = G + ridge * eye(size(G));
    end
    betaA = G \ (XA' * y);
end
